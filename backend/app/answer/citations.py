from collections.abc import Sequence


def attach_citation_labels(results: Sequence[dict]) -> list[dict]:
    labeled: list[dict] = []
    for index, result in enumerate(results, start=1):
        enriched = dict(result)
        enriched["citation_label"] = f"[{index}]"
        labeled.append(enriched)
    return labeled

