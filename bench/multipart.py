"""Detect JF-ICR questions that bundle several asks into one entry.

30% of the public set (76/253) does this; the participant test set does not
(0/50), which is worth remembering before reading across the two.

Kept deliberately free of imports from solve.py so both solve.py (which gates
its aggregation-rule prompt block on this) and decompose_experiment.py (which
samples by it) can use it without a circular import. The entry points take
question TEXT, not a built prompt -- prompt parsing stays with the caller.

The signal is not 「か」. Japanese IR questions are usually phrased as requests
(「〜を教えてほしい」「〜をお聞かせください」) rather than syntactic interrogatives,
so keying on a question particle alone misses most of them.
"""

import re

# Sentence-final request/interrogative markers.
ASK_MARKERS = [
    "ください", "下さい", "ほしい", "欲しい", "いただきたい", "頂きたい",
    "伺いたい", "伺います", "お伺い", "聞きたい", "知りたい", "教えて",
    "どうか", "いかがか", "考えか", "見解", "説明され", "答えられ",
    "なぜ", "どのよう", "どの程度", "どのくらい", "どれくらい", "いくら",
    "いつ", "どこ", "どちら", "どれ",
]

# A trailing 「か」/「の」 before the period is the other reliable signal.
ASK_TAIL = re.compile(r"(か|の)[。？?！!]?$")

# Discourse connectives that introduce an additional ask.
CONNECTIVES = [
    "また、", "また,", "あわせて", "併せて", "合わせて", "加えて",
    "さらに", "更に", "それから", "次に", "もう一点", "もう1点", "もう１点",
    "2点目", "２点目", "二点目", "1点目", "１点目", "一点目",
    "2点", "２点", "二点", "3点", "３点", "三点", "2つ", "２つ", "二つ",
    "3つ", "３つ", "三つ", "以下の点",
]

SENTENCE_SPLIT = re.compile(r"(?<=[。？?！!])\s*|\n+")


def sentences(text):
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s and s.strip()]


def is_ask(sentence):
    return any(m in sentence for m in ASK_MARKERS) or bool(ASK_TAIL.search(sentence))


def heuristic_multipart(question):
    """(is_multi, n_ask_sentences, connectives_found) for a question string."""
    sents = sentences(question)
    n_ask = sum(1 for s in sents if is_ask(s))
    found = [c for c in CONNECTIVES if c in question]
    is_multi = n_ask >= 2 or (bool(found) and len(sents) >= 2)
    return is_multi, n_ask, found


def is_multipart(question):
    """Just the boolean, for callers that only need to gate on it."""
    return heuristic_multipart(question)[0]
