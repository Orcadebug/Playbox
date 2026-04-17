from __future__ import annotations

from typing import Literal

from app.retrieval.bm25 import BM25Tokenizer
from app.retrieval.trie import Pattern


def build_query_patterns(
    query: str,
    max_patterns: int = 2000,
    phrase_ngram: int = 2,
    phrase_weight: float = 1.5,
) -> list[Pattern]:
    tokenizer = BM25Tokenizer(use_stemming=False, use_stopwords=False)
    tokens = tokenizer.tokenize(query)
    seen: set[tuple[str, str]] = set()
    patterns: list[Pattern] = []

    def add(body: str, weight: float, kind: Literal["token", "phrase"]) -> None:
        if len(patterns) >= max_patterns:
            return
        key = (kind, body)
        if not body or key in seen:
            return
        seen.add(key)
        patterns.append(
            Pattern(
                id=len(patterns),
                body=body,
                weight=weight,
                kind=kind,
            )
        )

    for token in tokens:
        add(token, 1.0, "token")

    if len(tokens) > 1:
        add(" ".join(tokens), 2.0, "phrase")

    if phrase_ngram > 1 and len(tokens) >= phrase_ngram:
        for start in range(0, len(tokens) - phrase_ngram + 1):
            add(" ".join(tokens[start : start + phrase_ngram]), phrase_weight, "phrase")

    return patterns
