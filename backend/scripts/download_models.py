"""Download the Waver ONNX cross-encoder reranker model and tokenizer files.

Usage:
    uv run python backend/scripts/download_models.py
    uv run python backend/scripts/download_models.py --quantize
    uv run python backend/scripts/download_models.py --output-dir backend/models/my-model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from urllib.request import urlopen

_log = logging.getLogger(__name__)

_HF_REPO = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/resolve/main"
_DEFAULT_OUTPUT_DIR = "models/cross-encoder-ms-marco-MiniLM-L-6-v2"

_FILES = {
    "model.onnx": f"{_HF_REPO}/onnx/model.onnx",
    "tokenizer.json": f"{_HF_REPO}/tokenizer.json",
    "special_tokens_map.json": f"{_HF_REPO}/special_tokens_map.json",
    "tokenizer_config.json": f"{_HF_REPO}/tokenizer_config.json",
}


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} → {output_path}")
    sha256 = hashlib.sha256()
    with urlopen(url) as response, output_path.open("wb") as handle:  # nosec: B310
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            sha256.update(chunk)
    digest = sha256.hexdigest()[:16]
    print(f"  ✓ {output_path.name} (sha256_prefix={digest})")
    return output_path


def quantize_model(model_path: Path) -> Path:
    """INT8 dynamic quantization — ~2x speedup on CPU, ~75% size reduction."""
    quantized_path = model_path.parent / "model_quantized.onnx"
    print(f"  Quantizing {model_path.name} → {quantized_path.name} …")
    try:
        from onnxruntime.quantization import (  # type: ignore[import-untyped]
            QuantType,
            quantize_dynamic,
        )

        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )
        print(f"  ✓ Quantized model saved to {quantized_path}")
        return quantized_path
    except ImportError:
        print("  ✗ onnxruntime.quantization not available — skipping quantization")
        return model_path


def sha256_prefix(path: Path) -> str | None:
    if not path.exists():
        return None
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def write_manifest(model_root: Path) -> None:
    required = {
        "projection": Path("models/projection.npz"),
        "reranker": model_root / "model_quantized.onnx",
        "reranker_tokenizer": model_root / "tokenizer.json",
        "mrl": Path("models/mrl/model.onnx"),
        "mrl_tokenizer": Path("models/mrl/tokenizer.json"),
    }
    manifest = {
        name: {
            "path": str(path),
            "present": path.exists(),
            "sha256_prefix": sha256_prefix(path),
        }
        for name, path in required.items()
    }
    manifest_path = Path("models/artifacts.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    missing = [name for name, data in manifest.items() if not data["present"]]
    print(f"\nWrote artifact manifest: {manifest_path}")
    if missing:
        print("Missing production artifacts:", ", ".join(missing))


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Download Waver ONNX reranker + tokenizer.")
    parser.add_argument(
        "--profile",
        choices=["reranker", "production"],
        default="reranker",
        help="Use 'production' to download reranker artifacts and verify the full artifact set.",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory to save model files (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize model to INT8 after download (~75%% size reduction, ~2x CPU speedup)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist (default: True)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"\nDownloading cross-encoder reranker to {output_dir}/\n")

    for filename, url in _FILES.items():
        dest = output_dir / filename
        if args.skip_existing and dest.exists():
            print(f"  → {filename} already exists, skipping")
            continue
        download_file(url, dest)

    if args.quantize:
        model_path = output_dir / "model.onnx"
        if model_path.exists():
            print()
            quantize_model(model_path)
        else:
            print("  ✗ model.onnx not found — cannot quantize")
            return 1

    if args.profile == "production":
        needs_quantized = not (output_dir / "model_quantized.onnx").exists()
        if needs_quantized and (output_dir / "model.onnx").exists():
            quantize_model(output_dir / "model.onnx")
        write_manifest(output_dir)

    print(f"\nDone. Model directory: {output_dir.resolve()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
