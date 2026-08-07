"""Non-LLM baseline (b): fine-tune a small Japanese encoder under k-fold CV.

Fine-tunes a pretrained Japanese BERT-family encoder (default:
cl-tohoku/bert-base-japanese-v3, ~110M params) with a 5-way classification
head on the company response text, fold by fold, and collects out-of-fold
predictions -- same protocol as bench/solve_lexicon.py and for the same
reason: the public set is only 253 rows with one label (-2) at just 2
examples, so a single train/test split would be too noisy, and stratified
CV isn't feasible with a 2-member class.

A fresh copy of the pretrained encoder is fine-tuned per fold (not
incrementally across folds) so no fold's predictions are contaminated by
having seen its own held-out rows during another fold's training.

Usage:
    python bench/solve_encoder.py --out outputs/predictions_encoder.jsonl
    python bench/solve_encoder.py --model-name cl-tohoku/bert-base-japanese-v3 \
        --epochs 8 --batch-size 8 --out outputs/predictions_encoder.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from solve import VALID_LABELS, split_query

RESPONSE_MARKER = "Company Response:"
LABEL2ID = {label: i for i, label in enumerate(VALID_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def extract_response(query: str) -> str:
    """Pull just the company's response text out of a dataset prompt."""
    _, middle, _ = split_query(query)
    i = middle.index(RESPONSE_MARKER) + len(RESPONSE_MARKER)
    return middle[i:].strip()


class EncodedDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_fold(
    model_name: str,
    train_texts: list[str],
    train_labels: list[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    class_weight: str | None,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(VALID_LABELS)).to(device)

    encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    loader = DataLoader(EncodedDataset(encodings, train_labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Inverse-frequency weights from this fold's train split only (no leakage from the held-out fold).
    # With 5 classes this heavily skewed (+1 is ~55% of the set), plain unweighted cross-entropy
    # collapses to always predicting +1 -- see the README note on this.
    loss_weight = None
    if class_weight == "balanced":
        counts = torch.bincount(torch.tensor(train_labels), minlength=len(VALID_LABELS)).float()
        loss_weight = (len(train_labels) / (len(VALID_LABELS) * counts.clamp(min=1))).to(device)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=loss_weight)
            loss.backward()
            optimizer.step()
    return tokenizer, model


@torch.no_grad()
def predict(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[int]:
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        encodings = tokenizer(chunk, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        encodings = {k: v.to(device) for k, v in encodings.items()}
        logits = model(**encodings).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="JF-ICR_public_set.parquet")
    parser.add_argument("--model-name", default="cl-tohoku/bert-base-japanese-v3")
    parser.add_argument("--out", default="outputs/predictions_encoder.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="only use the first N rows (smoke test)")
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="'balanced' uses inverse-frequency weighted cross-entropy loss (per fold) to counter label skew",
    )
    args = parser.parse_args()
    class_weight = None if args.class_weight == "none" else args.class_weight

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)
    texts = [extract_response(q) for q in df["query"]]
    golds = df["answer"].tolist()
    labels = [LABEL2ID[g] for g in golds]
    ids = df["id"].tolist()

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_pred: list[str | None] = [None] * len(df)

    for fold, (train_idx, test_idx) in enumerate(kf.split(texts)):
        print(f"fold {fold + 1}/{args.folds}: {len(train_idx)} train, {len(test_idx)} test")
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        tokenizer, model = train_fold(
            args.model_name,
            train_texts,
            train_labels,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.max_length,
            class_weight,
        )
        preds = predict(tokenizer, model, test_texts, device, args.batch_size, args.max_length)
        for idx, pred_id in zip(test_idx, preds):
            oof_pred[idx] = ID2LABEL[pred_id]

        del model, tokenizer
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(len(df)):
            record = {"id": int(ids[i]), "gold": golds[i], "prediction": oof_pred[i]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(df)} out-of-fold predictions to {out_path}")


if __name__ == "__main__":
    main()
