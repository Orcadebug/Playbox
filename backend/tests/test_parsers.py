from __future__ import annotations

from app.parsers.base import ParserDetector, build_default_parser_registry
from app.parsers.detector import detect_parser_name
from app.parsers.html import HTMLParser
from app.parsers.json_parser import JSONParser
from app.parsers.markdown import MarkdownParser
from app.parsers.pdf import PDFParser
from app.parsers.plaintext import PlainTextParser


def test_detect_parser_name_uses_content_sniffing_before_extension() -> None:
    assert detect_parser_name("notes.txt", b"%PDF-1.4\nfake") == "pdf"
    assert detect_parser_name("data.unknown", b"<html><body>Hi</body></html>") == "html"
    assert detect_parser_name("data.txt", b'{"ok": true}') == "json"


def test_parser_detector_routes_and_parses_common_formats() -> None:
    detector = ParserDetector(
        registry=build_default_parser_registry(), default_parser=PlainTextParser()
    )

    plain = detector.parse("notes.txt", b" hello world ")
    assert plain.parser_name == "plaintext"
    assert plain.content == " hello world "
    assert plain.documents[0].metadata["offset_basis"] == "source"

    markdown = MarkdownParser().parse("readme.md", b"# Title\n\n* item*")
    assert markdown.parser_name == "markdown"
    assert markdown.documents[0].content == "Title\n\nitem"

    csv_file = detector.parse("customers.csv", b"name,issue\nAlice,billing\nBob,shipping\n")
    assert csv_file.parser_name == "csv"
    assert [doc.content for doc in csv_file.documents] == [
        "name: Alice | issue: billing",
        "name: Bob | issue: shipping",
    ]

    json_file = JSONParser().parse("data.json", b'{"customer":"Acme","nested":{"plan":"pro"}}')
    assert json_file.parser_name == "json"
    assert "customer: Acme" in json_file.documents[0].content
    assert "nested.plan: pro" in json_file.documents[0].content

    html_file = HTMLParser().parse(
        "page.html",
        b"<html><head><style>.x{}</style><script>bad()</script></head><body><h1>Hello</h1><p>World</p></body></html>",
    )
    assert html_file.parser_name == "html"
    assert html_file.documents[0].content == "Hello World"
    assert "bad" not in html_file.documents[0].content

    pdf_file = PDFParser().parse("file.pdf", b"%PDF-1.4\nHello from pdf")
    assert pdf_file.parser_name == "pdf"
    assert pdf_file.documents
    assert "Hello from pdf" in pdf_file.documents[0].content
