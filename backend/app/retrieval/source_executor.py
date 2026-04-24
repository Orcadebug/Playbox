from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.parsers.base import (
    ParsedDocument,
    ParserDetector,
    SourceLocation,
    build_default_parser_registry,
)
from app.parsers.plaintext import PlainTextParser
from app.schemas.search import SearchDocument


def _default_parser_detector() -> ParserDetector:
    return ParserDetector(
        registry=build_default_parser_registry(),
        default_parser=PlainTextParser(),
    )


def _as_int(value: object, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _line_segments(text: str) -> list[tuple[str, int, int]]:
    if not text:
        return []

    segments: list[tuple[str, int, int]] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        clean_line = raw_line.rstrip("\r\n")
        start = cursor
        end = start + len(clean_line)
        cursor += len(raw_line)
        if clean_line:
            segments.append((clean_line, start, end))

    # splitlines() omits a final empty line when text ends with a newline.
    if not segments and text.strip():
        return [(text, 0, len(text))]
    return segments


@dataclass(slots=True)
class SourceRecord:
    record_id: str
    source_id: str
    source_name: str
    parser_name: str
    source_type: str
    source_origin: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    location: SourceLocation = field(default_factory=SourceLocation)
    source_start: int = 0
    source_end: int = 0


@dataclass(slots=True)
class SourceWindow:
    window_id: str
    record_id: str
    source_id: str
    source_name: str
    source_type: str
    source_origin: str
    parser_name: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    location: SourceLocation = field(default_factory=SourceLocation)
    source_start: int = 0
    source_end: int = 0
    neighboring_window_ids: list[str] = field(default_factory=list)


def build_source_records(
    documents: Sequence[SearchDocument],
    parser_detector: ParserDetector | None = None,
) -> list[SourceRecord]:
    detector = parser_detector or _default_parser_detector()
    records: list[SourceRecord] = []

    for source_index, document in enumerate(documents):
        source_metadata = dict(document.metadata or {})
        forced_parser_name = source_metadata.get("parser_name")
        parser = (
            detector.registry.get(str(forced_parser_name))
            if forced_parser_name is not None
            else None
        )
        if parser is None:
            parser = detector.detect(
                file_name=document.file_name,
                content=document.data,
                media_type=document.media_type,
            )
        parser_name = str(source_metadata.get("parser_name") or parser.name)
        source_name = document.file_name
        source_type = str(source_metadata.get("source_type") or "stored")
        source_origin = str(source_metadata.get("source_origin") or "stored")
        source_id = str(source_metadata.get("source_id") or f"source:{source_index}:{source_name}")

        parsed_documents: Sequence[ParsedDocument] = tuple(
            parser.iter_parse(
                file_name=document.file_name,
                content=document.data,
                media_type=document.media_type,
            )
        )
        for record_index, parsed_document in enumerate(parsed_documents):
            text = parsed_document.content
            if not text:
                continue

            metadata = dict(source_metadata)
            metadata.update(parsed_document.metadata)
            metadata.setdefault("parser_name", parser_name)
            if document.media_type is not None:
                metadata.setdefault("media_type", document.media_type)

            fallback_start = 0
            start = _as_int(
                metadata.get(
                    "source_start", metadata.get("char_start", metadata.get("byte_start"))
                ),
                fallback_start,
            )
            end = _as_int(
                metadata.get("source_end", metadata.get("char_end", metadata.get("byte_end"))),
                start + len(text),
            )
            end = max(start, end)

            record_id = f"{source_id}:record:{record_index}"
            records.append(
                SourceRecord(
                    record_id=record_id,
                    source_id=source_id,
                    source_name=source_name,
                    parser_name=parser_name,
                    source_type=source_type,
                    source_origin=source_origin,
                    text=text,
                    metadata=metadata,
                    location=parsed_document.location,
                    source_start=start,
                    source_end=end,
                )
            )

    return records


def build_source_windows(records: Sequence[SourceRecord]) -> list[SourceWindow]:
    windows: list[SourceWindow] = []

    for record in records:
        text = record.text
        if not text.strip():
            continue

        if record.parser_name == "plaintext":
            segments = _line_segments(text)
            if not segments:
                segments = [(text, 0, len(text))]
        else:
            segments = [(text, 0, len(text))]

        for segment_index, (segment_text, start, end) in enumerate(segments):
            if not segment_text.strip():
                continue
            window_metadata = dict(record.metadata)
            location = record.location
            if record.parser_name == "plaintext" and location.line_number is None:
                location = SourceLocation(
                    page_number=location.page_number,
                    row_number=location.row_number,
                    line_number=segment_index + 1,
                    section=location.section,
                )
            window_metadata.update(
                {
                    "record_id": record.record_id,
                    "window_index": segment_index,
                }
            )
            window_id = f"{record.record_id}:window:{segment_index}"
            windows.append(
                SourceWindow(
                    window_id=window_id,
                    record_id=record.record_id,
                    source_id=record.source_id,
                    source_name=record.source_name,
                    source_type=record.source_type,
                    source_origin=record.source_origin,
                    parser_name=record.parser_name,
                    text=segment_text,
                    metadata=window_metadata,
                    location=location,
                    source_start=record.source_start + start,
                    source_end=record.source_start + end,
                )
            )

    by_source: dict[str, list[SourceWindow]] = defaultdict(list)
    for window in windows:
        by_source[window.source_id].append(window)

    for source_windows in by_source.values():
        source_windows.sort(key=lambda item: (item.source_start, item.source_end, item.window_id))
        for index, window in enumerate(source_windows):
            neighbors: list[str] = []
            if index > 0:
                neighbors.append(source_windows[index - 1].window_id)
            if index + 1 < len(source_windows):
                neighbors.append(source_windows[index + 1].window_id)
            window.neighboring_window_ids = neighbors
            window.metadata["neighboring_window_ids"] = neighbors

    return windows


def build_source_windows_from_documents(
    documents: Sequence[SearchDocument],
    parser_detector: ParserDetector | None = None,
) -> list[SourceWindow]:
    records = build_source_records(documents, parser_detector=parser_detector)
    return build_source_windows(records)


__all__ = [
    "SourceRecord",
    "SourceWindow",
    "build_source_records",
    "build_source_windows",
    "build_source_windows_from_documents",
]
