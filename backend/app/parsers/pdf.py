from __future__ import annotations

from app.parsers.base import BaseParser, ParsedDocument, ParsedFile, SourceLocation


class PDFParser(BaseParser):
    name = "pdf"
    supported_extensions = frozenset({".pdf"})

    def parse(self, file_name: str, content: bytes, media_type: str | None = None) -> ParsedFile:
        raw = content
        text = self._extract_with_pymupdf(raw)
        if not text.strip():
            text = raw.decode("utf-8", errors="replace")
        pages = [page.strip() for page in text.split("\f") if page.strip()]
        if not pages:
            pages = [text.strip()] if text.strip() else []

        documents: list[ParsedDocument] = []
        for page_number, page_text in enumerate(pages, start=1):
            documents.append(
                ParsedDocument(
                    content=page_text,
                    source_name=file_name,
                    metadata={"page_number": page_number, "offset_basis": "parsed"},
                    location=SourceLocation(page_number=page_number),
                )
            )
        return ParsedFile(
            file_name=file_name, documents=documents, parser_name=self.name, media_type=media_type
        )

    def _extract_with_pymupdf(self, raw: bytes) -> str:
        try:
            import fitz  # type: ignore
        except Exception:
            return ""

        try:
            document = fitz.open(stream=raw, filetype="pdf")
        except Exception:
            return ""

        parts: list[str] = []
        try:
            for page in document:
                try:
                    parts.append(page.get_text("text"))
                except Exception:
                    parts.append("")
        finally:
            try:
                document.close()
            except Exception:
                pass
        return "\f".join(parts)
