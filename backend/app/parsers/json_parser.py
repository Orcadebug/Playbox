from __future__ import annotations

import json
from typing import Any

from app.parsers.base import BaseParser, ParsedDocument, ParsedFile


def _flatten_json(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json(item, child_prefix))
    elif isinstance(value, list):
        if not value:
            lines.append(f"{prefix}: []")
        else:
            for index, item in enumerate(value):
                child_prefix = f"{prefix}[{index}]"
                lines.extend(_flatten_json(item, child_prefix))
    else:
        lines.append(f"{prefix}: {value}" if prefix else str(value))
    return lines


class JSONParser(BaseParser):
    name = "json"
    supported_extensions = frozenset({".json", ".ndjson"})

    def parse(self, file_name: str, content: bytes, media_type: str | None = None) -> ParsedFile:
        text = content.decode("utf-8", errors="replace")
        stripped = text.strip()
        if not stripped:
            return ParsedFile(file_name=file_name, documents=[], parser_name=self.name, media_type=media_type)

        documents: list[ParsedDocument] = []
        if "\n" in stripped and all(line.strip().startswith("{") for line in stripped.splitlines() if line.strip()):
            for line_number, line in enumerate(stripped.splitlines(), start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                documents.append(
                    ParsedDocument(
                        content="\n".join(_flatten_json(payload)),
                        source_name=file_name,
                        metadata={"line_number": line_number},
                    )
                )
            return ParsedFile(file_name=file_name, documents=documents, parser_name=self.name, media_type=media_type)

        payload = json.loads(stripped)
        if isinstance(payload, list):
            for index, item in enumerate(payload):
                documents.append(
                    ParsedDocument(
                        content="\n".join(_flatten_json(item)),
                        source_name=file_name,
                        metadata={"item_index": index},
                    )
                )
            return ParsedFile(file_name=file_name, documents=documents, parser_name=self.name, media_type=media_type)

        documents.append(
            ParsedDocument(
                content="\n".join(_flatten_json(payload)),
                source_name=file_name,
            )
        )
        return ParsedFile(file_name=file_name, documents=documents, parser_name=self.name, media_type=media_type)
