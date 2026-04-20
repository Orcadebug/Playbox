from __future__ import annotations

import csv
import io

from app.parsers.base import BaseParser, ParsedDocument, ParsedFile, SourceLocation


def _line_offsets(text: str) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        cursor += len(line)
        offsets.append((start, cursor))
    return offsets


class CSVParser(BaseParser):
    name = "csv"
    supported_extensions = frozenset({".csv"})

    def parse(self, file_name: str, content: bytes, media_type: str | None = None) -> ParsedFile:
        text = content.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return ParsedFile(
                file_name=file_name, documents=[], parser_name=self.name, media_type=media_type
            )

        header = rows[0]
        offsets = _line_offsets(text)
        documents: list[ParsedDocument] = []
        for row_index, row in enumerate(rows[1:], start=2):
            if not any(cell.strip() for cell in row):
                continue
            parts: list[str] = []
            for column, value in zip(header, row, strict=False):
                value = value.strip()
                if value:
                    parts.append(f"{column.strip()}: {value}")
            documents.append(
                ParsedDocument(
                    content=" | ".join(parts),
                    source_name=file_name,
                    metadata={
                        "row_index": row_index,
                        "raw_source_start": offsets[row_index - 1][0]
                        if row_index - 1 < len(offsets)
                        else None,
                        "raw_source_end": offsets[row_index - 1][1]
                        if row_index - 1 < len(offsets)
                        else None,
                        "offset_basis": "parsed",
                    },
                    location=SourceLocation(row_number=row_index),
                )
            )
        return ParsedFile(
            file_name=file_name, documents=documents, parser_name=self.name, media_type=media_type
        )
