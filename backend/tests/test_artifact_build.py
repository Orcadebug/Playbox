from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from app.retrieval.sparse_projection import (
    DeterministicSemanticProjection,
    MrlProjection,
    ProjectionConfig,
    SparseProjection,
)
from scripts.build_artifacts_kaggle import _job_source
from scripts.train_projection import build_projection
from scripts.write_artifact_manifest import build_manifest


def test_projection_distillation_writes_sparse_projection(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "pack.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "case-1",
                        "query": "duplicate charge refund",
                        "documents": [
                            {
                                "source_id": "gold",
                                "content": (
                                    "Customer reports duplicate charge after checkout retry."
                                ),
                            },
                            {
                                "source_id": "other",
                                "content": "Shipping label printer settings only.",
                            },
                        ],
                        "expected": [{"source_id": "gold"}],
                    }
                ]
            }
        )
    )
    out = tmp_path / "projection.npz"

    report = build_projection(
        fixture_dirs=[fixtures],
        teacher="deterministic",
        mrl_model=None,
        out=out,
        report_out=tmp_path / "projection.report.json",
        hash_features=512,
        dim=16,
        ridge=0.01,
        synthetic_count=8,
        max_examples=None,
        validation_fraction=0.2,
        seed=7,
        batch_size=4,
        query_prefix="",
        document_prefix="",
    )

    loaded = SparseProjection.load(out)
    assert loaded.config.hash_features == 512
    assert loaded.config.dim == 16
    assert report.examples_used >= 3
    assert report.nnz > 0
    assert np.isfinite(report.validation_mean_cosine)


def test_mrl_projection_applies_task_prefix(  # type: ignore[no-untyped-def]
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_mrl_encode(
        model_path: str,
        texts: list[str],
        dim: int,
        is_query: bool,
    ) -> list[list[float]]:
        captured["model_path"] = model_path
        captured["texts"] = texts
        captured["is_query"] = is_query
        return [[1.0] * dim for _ in texts]

    monkeypatch.setattr("app.retrieval.sparse_projection.get_attr", lambda name: fake_mrl_encode)
    model_path = tmp_path / "mrl" / "model.onnx"
    model_path.parent.mkdir()
    model_path.write_bytes(b"placeholder")
    projection = MrlProjection(
        model_path=model_path,
        config=ProjectionConfig(hash_features=64, dim=4),
        fallback=DeterministicSemanticProjection(ProjectionConfig(hash_features=64, dim=4)),
        query_prefix="query: ",
        document_prefix="doc: ",
    )

    vector = projection.encode_query("billing refund")

    assert vector.shape == (4,)
    assert captured["texts"] == ["query: billing refund"]
    assert captured["is_query"] is True


def test_artifact_manifest_records_complete_bundle(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    projection = model_dir / "projection.npz"
    build_projection(
        fixture_dirs=[],
        teacher="deterministic",
        mrl_model=None,
        out=projection,
        report_out=model_dir / "projection.report.json",
        hash_features=128,
        dim=8,
        ridge=0.01,
        synthetic_count=12,
        max_examples=None,
        validation_fraction=0.2,
        seed=3,
        batch_size=4,
        query_prefix="",
        document_prefix="",
    )
    mrl_dir = model_dir / "mrl"
    mrl_dir.mkdir(parents=True)
    (mrl_dir / "model.onnx").write_bytes(b"mrl")
    (mrl_dir / "tokenizer.json").write_text("{}")
    (mrl_dir / "mrl.json").write_text(json.dumps({"model_id": "teacher"}))
    reranker_dir = model_dir / "cross-encoder-ms-marco-MiniLM-L-6-v2"
    reranker_dir.mkdir()
    (reranker_dir / "model_quantized.onnx").write_bytes(b"reranker")
    (reranker_dir / "tokenizer.json").write_text("{}")

    manifest = build_manifest(
        model_dir=model_dir,
        projection_path=projection,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_quantized=True,
        mrl_metadata=mrl_dir / "mrl.json",
        projection_report=model_dir / "projection.report.json",
        latency_report=None,
        eval_report=None,
    )

    assert manifest["complete"] is True
    assert manifest["missing"] == []
    assert manifest["artifacts"]["projection"]["dim"] == 8  # type: ignore[index]
    assert manifest["artifacts"]["mrl"]["model_id"] == "teacher"  # type: ignore[index]


def test_kaggle_job_source_runs_artifact_commands() -> None:
    source = _job_source(
        Namespace(
            model_id="mixedbread-ai/mxbai-embed-large-v1",
            exporter="torch",
            dim=256,
            hash_features=262144,
            synthetic_count=10,
            max_examples=20,
            source_dataset_slug="waver-artifact-source",
        )
    )

    assert "scripts/export_mrl_onnx.py" in source
    assert "scripts/train_projection.py" in source
    assert "waver-artifacts.tar.gz" in source
