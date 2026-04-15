from __future__ import annotations

import re

from app.parsers.base import BaseParser, ParsedDocument, ParsedFile


_MARKDOWN_PATTERNS = [
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"\*([^*]+)\*"), r"\1"),
    (re.compile(r"!\[([^\]]*)\]\([^)]+\)"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),
]


class MarkdownParser(BaseParser):
    name = "markdown"
    supported_extensions = frozenset({".md", ".markdown", ".mkd"})

    def parse(self, file_name: str, content: bytes, media_type: str | None = None) -> ParsedFile:
        text = content.decode("utf-8", errors="replace")
        cleaned = text
        for pattern, replacement in _MARKDOWN_PATTERNS:
            cleaned = pattern.sub(replacement, cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = "\n".join(line.strip() for line in cleaned.splitlines()).strip()
        document = ParsedDocument(content=cleaned, source_name=file_name)
        return ParsedFile(
            file_name=file_name,
            documents=[document] if document.content else [],
            parser_name=self.name,
            media_type=media_type,
        )
