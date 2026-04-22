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


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        cursor += len(line)
        spans.append((start, cursor, line))
    return spans


def _top_level_list_spans(text: str) -> list[tuple[int, int]]:
    decoder = json.JSONDecoder()
    start = text.find("[")
    if start < 0:
        return []

    cursor = start + 1
    spans: list[tuple[int, int]] = []
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] == "]":
            break
        item_start = cursor
        try:
            _, item_end = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            break
        spans.append((item_start, item_end))
        cursor = item_end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor < len(text) and text[cursor] == ",":
            cursor += 1
    return spans


class JSONParser(BaseParser):
    name = "json"
    supported_extensions = frozenset({".json", ".ndjson"})

    def parse(self, file_name: str, content: bytes, media_type: str | None = None) -> ParsedFile:
        text = content.decode("utf-8", errors="replace")
        stripped = text.strip()
        if not stripped:
            return ParsedFile(
                file_name=file_name, documents=[], parser_name=self.name, media_type=media_type
            )

        documents: list[ParsedDocument] = []
        if "\n" in stripped and all(
            line.strip().startswith("{") for line in stripped.splitlines() if line.strip()
        ):
            for line_number, (start, end, line) in enumerate(_line_spans(text), start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                documents.append(
                    ParsedDocument(
                        content="\n".join(_flatten_json(payload)),
                        source_name=file_name,
                        metadata={
                            "line_number": line_number,
                            "raw_source_start": start,
                            "raw_source_end": end,
                            "offset_basis": "parsed",
                        },
                    )
                )
            return ParsedFile(
                file_name=file_name,
                documents=documents,
                parser_name=self.name,
                media_type=media_type,
            )

        payload = json.loads(stripped)
        if isinstance(payload, list):
            item_spans = _top_level_list_spans(text)
            for index, item in enumerate(payload):
                raw_start, raw_end = item_spans[index] if index < len(item_spans) else (None, None)
                documents.append(
                    ParsedDocument(
                        content="\n".join(_flatten_json(item)),
                        source_name=file_name,
                        metadata={
                            "item_index": index,
                            "raw_source_start": raw_start,
                            "raw_source_end": raw_end,
                            "offset_basis": "parsed",
                        },
                    )
                )
            return ParsedFile(
                file_name=file_name,
                documents=documents,
                parser_name=self.name,
                media_type=media_type,
            )

        documents.append(
            ParsedDocument(
                content="\n".join(_flatten_json(payload)),
                source_name=file_name,
                metadata={
                    "raw_source_start": text.find(stripped),
                    "raw_source_end": text.find(stripped) + len(stripped),
                    "offset_basis": "parsed",
                },
            )
        )
        return ParsedFile(
            file_name=file_name, documents=documents, parser_name=self.name, media_type=media_type
        )
