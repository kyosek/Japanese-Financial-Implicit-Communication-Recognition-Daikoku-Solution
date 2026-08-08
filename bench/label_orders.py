"""Permutations of the label listing order for the label-order probe in
bench/prompt_sensitivity.py (--probe label-order).

Every row's prompt fixes the same instruction block naming the 5 valid
labels twice: an inline set `{"+2", "+1", "0", "-1", "-2"}` and a 5-line
"label meanings" block underneath it. `permute_label_order` rewrites both to
list the labels in a different order while keeping every label's meaning
text attached to it -- only the *position* each label is listed in changes,
not what it means or which token score_label() teacher-forces against, so
any resulting prediction shift is attributable to list order alone.
"""

from solve import VALID_LABELS

LABEL_MEANINGS = {
    "+2": "Strong Commitment",
    "+1": "Weak or Qualified Commitment",
    "0": "Neutral or Hedged Intent",
    "-1": "Weak Refusal",
    "-2": "Strong Refusal",
}

LABEL_ORDERS = {
    "original": ["+2", "+1", "0", "-1", "-2"],
    "reversed": ["-2", "-1", "0", "+1", "+2"],
    "neutral_first": ["0", "+2", "-2", "+1", "-1"],
    "extremes_first": ["+2", "-2", "+1", "-1", "0"],
    "shuffled": ["-1", "0", "+2", "-2", "+1"],
}


def _format_meanings_block(order: list[str]) -> str:
    lines = []
    for label in order:
        quoted = f'"{label}"'
        lines.append(" " * (8 - len(quoted)) + quoted + ' : "' + LABEL_MEANINGS[label] + '"')
    return "\n".join(lines)


_ORIGINAL_SET_STR = "{" + ", ".join(f'"{l}"' for l in VALID_LABELS) + "}"
_ORIGINAL_MEANINGS_BLOCK = _format_meanings_block(VALID_LABELS)


def permute_label_order(query: str, order: list[str]) -> str:
    """Rewrite a dataset prompt's label-set string and meanings block into `order`."""
    if sorted(order) != sorted(VALID_LABELS):
        raise ValueError(f"order must be a permutation of {VALID_LABELS}, got {order}")
    if _ORIGINAL_SET_STR not in query or _ORIGINAL_MEANINGS_BLOCK not in query:
        raise ValueError("query does not contain the expected fixed label-listing block -- dataset format changed?")
    new_set_str = "{" + ", ".join(f'"{l}"' for l in order) + "}"
    new_block = _format_meanings_block(order)
    return query.replace(_ORIGINAL_SET_STR, new_set_str).replace(_ORIGINAL_MEANINGS_BLOCK, new_block)
