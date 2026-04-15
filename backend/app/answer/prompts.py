from collections.abc import Sequence


def build_answer_messages(query: str, results: Sequence[dict]) -> list[dict[str, str]]:
    context_blocks = []
    for index, result in enumerate(results, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] Source: {result['source_name']}",
                    f"Title: {result.get('title') or 'Untitled'}",
                    f"Metadata: {result.get('metadata') or {}}",
                    result["content"],
                ]
            )
        )

    context = "\n\n".join(context_blocks)
    system = (
        "You answer only from the provided context. "
        "If the context is insufficient, say so plainly. "
        "Use citation markers like [1] [2] inline with your claims. "
        "Return concise markdown."
    )
    user = f"Question: {query}\n\nContext:\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

