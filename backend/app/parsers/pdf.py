from __future__ import annotations

from collections.abc import Iterator

from app.parsers.base import BaseParser, ParsedDocument, ParsedFile, SourceLocation


class PDFParser(BaseParser):
    name = "pdf"
    supported_extensions = frozenset({".pdf"})

    def parse(self, file_name: str, content: bytes, media_type: str | None = None) -> ParsedFile:
        documents = list(self.iter_parse(file_name, content, media_type))
        if not documents and content.strip():
            documents = [
                ParsedDocument(
                    content=content.decode("utf-8", errors="replace"),
                    source_name=file_name,
                    metadata={"offset_basis": "parsed"},
                    location=SourceLocation(),
                )
            ]
        return ParsedFile(
            file_name=file_name, documents=documents, parser_name=self.name, media_type=media_type
        )

    def iter_parse(
        self,
        file_name: str,
        content: bytes,
        media_type: str | None = None,
    ) -> Iterator[ParsedDocument]:
        emitted = False
        for page_number, page_text in self._iter_with_pdfium(content):
            clean_text = page_text.strip()
            if not clean_text:
                continue
            emitted = True
            yield ParsedDocument(
                content=clean_text,
                source_name=file_name,
                metadata={"page_number": page_number, "offset_basis": "parsed"},
                location=SourceLocation(page_number=page_number),
            )

        if emitted:
            return

        page_texts = self._extract_with_pymupdf(content).split("\f")
        for page_number, page_text in enumerate(page_texts, start=1):
            clean_text = page_text.strip()
            if not clean_text:
                continue
            emitted = True
            yield ParsedDocument(
                content=clean_text,
                source_name=file_name,
                metadata={"page_number": page_number, "offset_basis": "parsed"},
                location=SourceLocation(page_number=page_number),
            )

        if not emitted and content.strip():
            yield ParsedDocument(
                content=content.decode("utf-8", errors="replace"),
                source_name=file_name,
                metadata={"offset_basis": "parsed"},
                location=SourceLocation(),
            )

    def _iter_with_pdfium(self, raw: bytes) -> Iterator[tuple[int, str]]:
        try:
            import pypdfium2 as pdfium  # type: ignore[import-untyped]
        except Exception:
            return

        document = None
        try:
            document = pdfium.PdfDocument(raw)
            for page_index in range(len(document)):
                try:
                    textpage = document[page_index].get_textpage()
                    try:
                        yield page_index + 1, textpage.get_text_range()
                    finally:
                        textpage.close()
                except Exception:
                    yield page_index + 1, ""
        except Exception:
            return
        finally:
            if document is not None:
                try:
                    document.close()
                except Exception:
                    pass

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
