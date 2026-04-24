"""Download the Waver ONNX cross-encoder reranker model and tokenizer files.

Usage:
    uv run python backend/scripts/download_models.py
    uv run python backend/scripts/download_models.py --quantize
    uv run python backend/scripts/download_models.py --output-dir backend/models/my-model
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from urllib.request import urlopen

_log = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main"
_DEFAULT_OUTPUT_DIR = "backend/models/cross-encoder-ms-marco-MiniLM-L-6-v2"

_FILES = {
    "model.onnx": f"{_HF_BASE}/model.onnx",
    "tokenizer.json": f"{_HF_BASE}/tokenizer.json",
    "special_tokens_map.json": f"{_HF_BASE}/special_tokens_map.json",
    "tokenizer_config.json": f"{_HF_BASE}/tokenizer_config.json",
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


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Download Waver ONNX reranker + tokenizer.")
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

    print(f"\nDone. Model directory: {output_dir.resolve()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
