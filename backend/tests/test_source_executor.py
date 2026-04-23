from __future__ import annotations

from app.retrieval.source_executor import (
    build_source_records,
    build_source_windows,
    build_source_windows_from_documents,
)
from app.schemas.search import SearchDocument


def test_plaintext_windows_split_by_line_with_exact_offsets() -> None:
    content = "alpha line\nbeta line\n\ngamma line"
    documents = [
        SearchDocument(
            file_name="notes.txt",
            content=content,
            media_type="text/plain",
            metadata={"source_id": "raw-1", "source_origin": "raw", "source_type": "raw"},
        )
    ]

    windows = build_source_windows_from_documents(documents)
    assert [window.text for window in windows] == ["alpha line", "beta line", "gamma line"]
    assert content[windows[0].source_start : windows[0].source_end] == "alpha line"
    assert content[windows[1].source_start : windows[1].source_end] == "beta line"
    assert content[windows[2].source_start : windows[2].source_end] == "gamma line"
    assert windows[0].neighboring_window_ids == [windows[1].window_id]
    assert windows[1].neighboring_window_ids == [windows[0].window_id, windows[2].window_id]


def test_csv_rows_become_record_windows_with_row_locations() -> None:
    documents = [
        SearchDocument(
            file_name="complaints.csv",
            content=(
                b"customer,issue\n"
                b"Acme,billing complaint\n"
                b"Beta,shipping delay\n"
            ),
            media_type="text/csv",
            metadata={"source_id": "csv-1", "source_origin": "stored", "source_type": "upload"},
        )
    ]

    records = build_source_records(documents)
    windows = build_source_windows(records)
    assert len(records) == 2
    assert len(windows) == 2
    assert windows[0].location.row_number == 2
    assert windows[1].location.row_number == 3
    assert "billing complaint" in windows[0].text


def test_json_list_becomes_logical_record_windows() -> None:
    json_payload = '[{"ticket":"A","issue":"timeout"},{"ticket":"B","issue":"refund"}]'
    documents = [
        SearchDocument(
            file_name="events.json",
            content=json_payload,
            media_type="application/json",
            metadata={
                "source_id": "json-1",
                "source_origin": "connector",
                "source_type": "webhook",
            },
        )
    ]

    windows = build_source_windows_from_documents(documents)
    assert len(windows) == 2
    assert windows[0].metadata["item_index"] == 0
    assert windows[1].metadata["item_index"] == 1
    assert "ticket: A" in windows[0].text
    assert "ticket: B" in windows[1].text


def test_empty_source_produces_no_windows() -> None:
    documents = [
        SearchDocument(
            file_name="empty.txt",
            content="",
            media_type="text/plain",
            metadata={"source_id": "empty"},
        )
    ]

    assert build_source_windows_from_documents(documents) == []
