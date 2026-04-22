from __future__ import annotations

from html.parser import HTMLParser as _HTMLParser

from app.parsers.base import BaseParser, ParsedDocument, ParsedFile


class _TextExtractor(_HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self._parts)


class HTMLParser(BaseParser):
    name = "html"
    supported_extensions = frozenset({".html", ".htm"})

    def parse(self, file_name: str, content: bytes, media_type: str | None = None) -> ParsedFile:
        text = content.decode("utf-8", errors="replace")
        extractor = _TextExtractor()
        extractor.feed(text)
        extractor.close()
        document = ParsedDocument(
            content=extractor.text().strip(),
            source_name=file_name,
            metadata={
                "raw_source_start": 0,
                "raw_source_end": len(text),
                "offset_basis": "parsed",
            },
        )
        return ParsedFile(
            file_name=file_name,
            documents=[document] if document.content else [],
            parser_name=self.name,
            media_type=media_type,
        )
