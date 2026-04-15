from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(slots=True)
class SourceLocation:
    page_number: int | None = None
    row_number: int | None = None
    line_number: int | None = None
    section: str | None = None


@dataclass(slots=True)
class ParsedDocument:
    content: str
    source_name: str
    metadata: dict[str, object] = field(default_factory=dict)
    location: SourceLocation = field(default_factory=SourceLocation)

    @property
    def text(self) -> str:
        return self.content


@dataclass(slots=True)
class ParsedFile:
    file_name: str
    documents: list[ParsedDocument]
    parser_name: str
    media_type: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return "\n\n".join(document.content for document in self.documents if document.content)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    content: str
    source_name: str
    metadata: dict[str, object] = field(default_factory=dict)
    location: SourceLocation = field(default_factory=SourceLocation)
    token_count: int = 0

    @property
    def text(self) -> str:
        return self.content


@dataclass(slots=True)
class SourceInput:
    file_name: str
    content: bytes
    media_type: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class ParserError(ValueError):
    pass


class BaseParser(ABC):
    name: str
    supported_extensions: frozenset[str] = frozenset()

    @abstractmethod
    def parse(self, file_name: str, content: bytes, media_type: str | None = None) -> ParsedFile:
        raise NotImplementedError


@dataclass(slots=True)
class ParserDetector:
    registry: Mapping[str, BaseParser]
    default_parser: BaseParser

    def detect(self, file_name: str, content: bytes, media_type: str | None = None) -> BaseParser:
        from app.parsers.detector import detect_parser_name

        parser_name = detect_parser_name(file_name=file_name, content=content, media_type=media_type)
        parser = self.registry.get(parser_name)
        if parser is not None:
            return parser
        return self.default_parser

    def parse(self, file_name: str, content: bytes, media_type: str | None = None) -> ParsedFile:
        return self.detect(file_name, content, media_type).parse(file_name, content, media_type)


def build_default_parser_registry() -> dict[str, BaseParser]:
    from app.parsers.csv_parser import CSVParser
    from app.parsers.html import HTMLParser
    from app.parsers.json_parser import JSONParser
    from app.parsers.markdown import MarkdownParser
    from app.parsers.pdf import PDFParser
    from app.parsers.plaintext import PlainTextParser

    parsers: list[BaseParser] = [
        PlainTextParser(),
        MarkdownParser(),
        CSVParser(),
        JSONParser(),
        HTMLParser(),
        PDFParser(),
    ]
    return {parser.name: parser for parser in parsers}


__all__ = [
    "BaseParser",
    "ParsedDocument",
    "ParsedFile",
    "Chunk",
    "ParserDetector",
    "ParserError",
    "SourceInput",
    "SourceLocation",
    "build_default_parser_registry",
]
