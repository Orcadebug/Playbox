from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import subprocess
from pathlib import Path
from typing import Literal

ExportBackend = Literal["auto", "optimum", "torch"]

DEFAULT_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"


def sha256_prefix(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def _missing_dependency_message() -> str:
    return (
        "Missing training/export dependencies. Run this script with:\n"
        "  uv run --with torch --with transformers --with optimum[onnxruntime] "
        "python scripts/export_mrl_onnx.py\n"
    )


def _run_optimum_export(
    *,
    model_id: str,
    output_dir: Path,
    opset: int,
    trust_remote_code: bool,
) -> None:
    optimum = shutil.which("optimum-cli")
    if optimum is None:
        raise RuntimeError("optimum-cli is not installed")
    command = [
        optimum,
        "export",
        "onnx",
        "--model",
        model_id,
        "--task",
        "feature-extraction",
        "--opset",
        str(opset),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    command.append(str(output_dir))
    subprocess.run(command, check=True)
    model_path = output_dir / "model.onnx"
    if not model_path.exists():
        candidates = sorted(output_dir.glob("*.onnx"))
        if not candidates:
            raise RuntimeError("Optimum export completed but did not produce an ONNX file")
        candidates[0].replace(model_path)


def _load_transformers():
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(_missing_dependency_message()) from exc
    return torch, F, AutoModel, AutoTokenizer


def _run_torch_export(
    *,
    model_id: str,
    tokenizer_model: str,
    output_dir: Path,
    opset: int,
    max_length: int,
    trust_remote_code: bool,
) -> None:
    torch, F, AutoModel, AutoTokenizer = _load_transformers()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model.eval()
    model_forward = inspect.signature(model.forward)
    accepts_token_type_ids = "token_type_ids" in model_forward.parameters

    class MeanPooledEmbeddingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = model

        def forward(  # type: ignore[override]
            self,
            input_ids,
            attention_mask,
            token_type_ids=None,
        ):
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if accepts_token_type_ids and token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            output = self.inner(**kwargs)
            token_embeddings = getattr(output, "last_hidden_state", output[0])
            mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return F.normalize(pooled, p=2, dim=1)

    encoded = tokenizer(
        [
            "Represent this sentence for searching relevant passages: duplicate billing refund",
            "Ticket note: customer was billed twice after checkout retry.",
        ],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])

    wrapper = MeanPooledEmbeddingModel()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.onnx"
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "token_type_ids": {0: "batch", 1: "sequence"},
        "sentence_embedding": {0: "batch"},
    }
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["token_type_ids"],
            ),
            str(model_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["sentence_embedding"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )
    tokenizer.save_pretrained(output_dir)


def _ensure_tokenizer(
    *,
    output_dir: Path,
    tokenizer_model: str,
    trust_remote_code: bool,
) -> None:
    tokenizer_path = output_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(_missing_dependency_message()) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(output_dir)
    if not tokenizer_path.exists():
        raise RuntimeError(f"Tokenizer export did not create {tokenizer_path}")


def _validate_onnx(output_dir: Path) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime, numpy, and tokenizers are required to validate the exported model"
        ) from exc

    tokenizer = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=512)
    encoded = tokenizer.encode_batch(["duplicate billing refund", "checkout retry double charge"])
    seq_len = max(len(item.ids) for item in encoded)
    pad_id = tokenizer.token_to_id("[PAD]") or 0
    input_ids = np.asarray(
        [item.ids + [pad_id] * (seq_len - len(item.ids)) for item in encoded],
        dtype=np.int64,
    )
    attention_mask = np.asarray(
        [item.attention_mask + [0] * (seq_len - len(item.attention_mask)) for item in encoded],
        dtype=np.int64,
    )
    token_type_ids = np.zeros_like(input_ids)
    session = ort.InferenceSession(
        str(output_dir / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    input_names = {item.name for item in session.get_inputs()}
    if "token_type_ids" in input_names:
        inputs["token_type_ids"] = token_type_ids
    output = session.run(None, inputs)[0]
    if output.ndim not in {2, 3} or output.shape[0] != 2:
        raise RuntimeError(f"Unexpected MRL ONNX output shape: {output.shape}")
    if output.ndim == 3 and output.shape[1] < 1:
        raise RuntimeError(f"Unexpected empty token dimension in MRL ONNX output: {output.shape}")


def export_mrl_onnx(
    *,
    model_id: str,
    tokenizer_model: str,
    output_dir: Path,
    exporter: ExportBackend,
    opset: int,
    max_length: int,
    trust_remote_code: bool,
    validate: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    used_exporter = exporter
    if exporter in {"auto", "optimum"}:
        try:
            _run_optimum_export(
                model_id=model_id,
                output_dir=output_dir,
                opset=opset,
                trust_remote_code=trust_remote_code,
            )
            used_exporter = "optimum"
        except Exception:
            if exporter == "optimum":
                raise
            used_exporter = "torch"
            _run_torch_export(
                model_id=model_id,
                tokenizer_model=tokenizer_model,
                output_dir=output_dir,
                opset=opset,
                max_length=max_length,
                trust_remote_code=trust_remote_code,
            )
    else:
        _run_torch_export(
            model_id=model_id,
            tokenizer_model=tokenizer_model,
            output_dir=output_dir,
            opset=opset,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )
    _ensure_tokenizer(
        output_dir=output_dir,
        tokenizer_model=tokenizer_model,
        trust_remote_code=trust_remote_code,
    )
    if validate:
        _validate_onnx(output_dir)

    model_path = output_dir / "model.onnx"
    tokenizer_path = output_dir / "tokenizer.json"
    metadata: dict[str, object] = {
        "artifact": "mrl",
        "model_id": model_id,
        "tokenizer_model": tokenizer_model,
        "exporter": used_exporter,
        "opset": opset,
        "max_length": max_length,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "model_sha256_prefix": sha256_prefix(model_path),
        "tokenizer_sha256_prefix": sha256_prefix(tokenizer_path),
    }
    (output_dir / "mrl.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Waver's MRL embedding model to ONNX.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models/mrl"))
    parser.add_argument("--exporter", choices=["auto", "optimum", "torch"], default="auto")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer_model = args.tokenizer_model or args.model_id
    metadata = export_mrl_onnx(
        model_id=args.model_id,
        tokenizer_model=tokenizer_model,
        output_dir=args.output_dir,
        exporter=args.exporter,
        opset=args.opset,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        validate=not args.no_validate,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
