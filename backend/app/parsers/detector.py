from __future__ import annotations

import csv
import json
import mimetypes
import re
from pathlib import Path

from app.parsers.base import BaseParser, ParserDetector, build_default_parser_registry
from app.parsers.plaintext import PlainTextParser


def _decode_sample(content: bytes, limit: int = 4096) -> str:
    return content[:limit].decode("utf-8", errors="replace")


def detect_parser_name(file_name: str, content: bytes, media_type: str | None = None) -> str:
    sample = _decode_sample(content).strip()
    if content.startswith(b"%PDF-"):
        return "pdf"
    if re.search(r"<\s*html|<!doctype html", sample, re.IGNORECASE):
        return "html"
    if sample.startswith("{") or sample.startswith("["):
        try:
            json.loads(sample)
            return "json"
        except Exception:
            pass
    if "\n" in sample and "," in sample:
        try:
            dialect = csv.Sniffer().sniff(sample)
            if dialect.delimiter:
                return "csv"
        except Exception:
            pass

    suffix = Path(file_name).suffix.lower()
    extension_map = {
        ".txt": "plaintext",
        ".log": "plaintext",
        ".text": "plaintext",
        ".md": "markdown",
        ".markdown": "markdown",
        ".mkd": "markdown",
        ".csv": "csv",
        ".json": "json",
        ".ndjson": "json",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf",
    }
    if suffix in extension_map:
        return extension_map[suffix]

    if media_type:
        normalized = media_type.split(";", 1)[0].strip().lower()
        media_map = {
            "application/pdf": "pdf",
            "text/html": "html",
            "text/csv": "csv",
            "application/csv": "csv",
            "application/json": "json",
            "text/markdown": "markdown",
            "text/plain": "plaintext",
        }
        if normalized in media_map:
            return media_map[normalized]

    if file_name:
        mime, _ = mimetypes.guess_type(file_name)
        if mime:
            guessed = {
                "application/pdf": "pdf",
                "text/html": "html",
                "text/csv": "csv",
                "application/csv": "csv",
                "application/json": "json",
                "text/plain": "plaintext",
            }.get(mime)
            if guessed:
                return guessed
    return "plaintext"


def detect_parser(file_name: str, content: bytes, media_type: str | None = None) -> BaseParser:
    registry = build_default_parser_registry()
    detector = ParserDetector(registry=registry, default_parser=PlainTextParser())
    return detector.detect(file_name, content, media_type)


def detect_and_parse(file_name: str, content: bytes, media_type: str | None = None):
    parser = detect_parser(file_name=file_name, content=content, media_type=media_type)
    return parser.parse(file_name=file_name, content=content, media_type=media_type)
