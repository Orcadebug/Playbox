from __future__ import annotations

from app.parsers.base import BaseParser, ParsedDocument, ParsedFile


class PlainTextParser(BaseParser):
    name = "plaintext"
    supported_extensions = frozenset({".txt", ".log", ".text"})

    def parse(self, file_name: str, content: bytes, media_type: str | None = None) -> ParsedFile:
        text = content.decode("utf-8", errors="replace")
        document = ParsedDocument(
            content=text,
            source_name=file_name,
            metadata={
                "source_start": 0,
                "source_end": len(text),
                "char_start": 0,
                "char_end": len(text),
                "byte_start": 0,
                "byte_end": len(text.encode("utf-8")),
                "offset_basis": "source",
            },
        )
        return ParsedFile(
            file_name=file_name,
            documents=[document] if text.strip() else [],
            parser_name=self.name,
            media_type=media_type,
        )
