from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


DEFAULT_MODEL_URL = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/model.onnx"


def download_model(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, output_path.open("wb") as handle:  # nosec: B310
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Waver ONNX reranker model.")
    parser.add_argument("--url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--output", default="backend/models/cross-encoder-ms-marco-MiniLM-L-6-v2.onnx")
    args = parser.parse_args()

    path = download_model(args.url, Path(args.output))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

