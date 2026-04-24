from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from pathlib import Path

DEFAULT_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_SOURCE_DATASET_SLUG = "waver-artifact-source"


def _kaggle_env_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.env"


def _kaggle_config_path() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _kaggle_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(_read_env_file(_kaggle_env_path()))
    return env


def _kaggle_username(env: dict[str, str]) -> str:
    username = str(env.get("KAGGLE_USERNAME") or "").strip()
    if username:
        return username
    config_path = _kaggle_config_path()
    if config_path.exists():
        payload = json.loads(config_path.read_text())
        username = str(payload.get("username") or "").strip()
        if username:
            return username
    if env.get("KAGGLE_API_TOKEN"):
        raise FileNotFoundError(
            "KAGGLE_API_TOKEN is configured, but KAGGLE_USERNAME is missing. "
            f"Add KAGGLE_USERNAME to {_kaggle_env_path()} or pass it in the environment."
        )
    raise FileNotFoundError(
        f"No Kaggle credentials found. Create {_kaggle_config_path()} or {_kaggle_env_path()}."
    )


def _backend_ignore(directory: str, names: list[str]) -> set[str]:
    del directory
    blocked = {
        ".env",
        ".env.local",
        ".env.production",
        ".envrc",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        "models",
        "target",
        "waver.db",
    }
    secret_suffixes = (".key", ".pem", ".p12", ".pfx")
    return {
        name
        for name in names
        if name in blocked
        or name.endswith(".pyc")
        or name.endswith(secret_suffixes)
        or "secret" in name.lower()
    }


def _copy_backend(source: Path, target: Path) -> None:
    shutil.copytree(source, target, ignore=_backend_ignore)


def _create_backend_archive(source: Path, archive_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="waver-backend-archive-") as tmp_name:
        tmp = Path(tmp_name)
        staged = tmp / "backend"
        _copy_backend(source, staged)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staged, arcname="backend")


def _prepare_source_dataset(args: argparse.Namespace, username: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "backend"
    workdir = Path(tempfile.mkdtemp(prefix="waver-kaggle-dataset-"))
    _create_backend_archive(backend_root, workdir / "backend.tar.gz")
    metadata = {
        "id": f"{username}/{args.source_dataset_slug}",
        "title": "Waver Artifact Source",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (workdir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    return workdir


def _sync_source_dataset(args: argparse.Namespace, username: str, env: dict[str, str]) -> None:
    dataset_dir = _prepare_source_dataset(args, username)
    dataset_id = f"{username}/{args.source_dataset_slug}"
    try:
        create = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(dataset_dir), "-r", "skip"],
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )
        if create.returncode == 0:
            print(f"Created private Kaggle source dataset {dataset_id}", flush=True)
            _wait_for_dataset_ready(dataset_id, env=env)
            return
        combined = f"{create.stdout}\n{create.stderr}".lower()
        if "already exists" not in combined and "dataset slug is already taken" not in combined:
            raise RuntimeError(create.stdout + create.stderr)
        version = subprocess.run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(dataset_dir),
                "-m",
                "Update Waver artifact source bundle",
                "-r",
                "skip",
                "-d",
            ],
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )
        if version.returncode != 0:
            raise RuntimeError(version.stdout + version.stderr)
        print(f"Updated private Kaggle source dataset {dataset_id}", flush=True)
        _wait_for_dataset_ready(dataset_id, env=env)
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)


def _wait_for_dataset_ready(
    dataset_id: str,
    *,
    env: dict[str, str],
    timeout_seconds: int = 600,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status = ""
    while time.monotonic() < deadline:
        status = subprocess.run(
            ["kaggle", "datasets", "status", dataset_id],
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )
        last_status = (status.stdout or status.stderr).strip()
        print(f"{dataset_id} dataset status: {last_status}", flush=True)
        if status.returncode == 0 and last_status.lower() == "ready":
            return
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for dataset {dataset_id}: {last_status}")


def _job_source(args: argparse.Namespace) -> str:
    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import json
        import os
        import shutil
        import subprocess
        import sys
        import tarfile
        from pathlib import Path

        ROOT = Path("/kaggle/working")
        INPUT_ROOT = Path("/kaggle/input")
        BACKEND = ROOT / "backend"

        def run(command: list[str], *, cwd: Path | None = None) -> None:
            print("$", " ".join(command), flush=True)
            subprocess.run(command, cwd=str(cwd or ROOT), check=True)

        def main() -> int:
            packages = [
                "numpy>=2.2.0,<2.4",
                "scipy>=1.14.0",
                "scikit-learn>=1.5.0",
                "tokenizers>=0.21.0",
                "onnxruntime>=1.20.1",
                "transformers>=4.40.0",
                "torch",
                "optimum[onnxruntime]",
            ]
            run([sys.executable, "-m", "pip", "install", "-q", *packages])
            if not BACKEND.exists():
                matches = sorted(INPUT_ROOT.rglob("backend.tar.gz"))
                if matches:
                    with tarfile.open(matches[0], "r:gz") as tar:
                        tar.extractall(ROOT)
                else:
                    backend_dirs = sorted(
                        path
                        for path in INPUT_ROOT.rglob("backend")
                        if (path / "pyproject.toml").exists()
                    )
                    if backend_dirs:
                        shutil.copytree(backend_dirs[0], BACKEND)
                    else:
                        available = [str(path) for path in INPUT_ROOT.rglob("*")]
                        raise FileNotFoundError(
                            {{"wanted": "backend source", "available": available[:50]}}
                        )
                if not BACKEND.exists():
                    available = [str(path) for path in INPUT_ROOT.rglob("*")]
                    raise FileNotFoundError(
                        {{"wanted": str(BACKEND), "available": available[:50]}}
                    )
            os.chdir(BACKEND)
            run([
                sys.executable,
                "scripts/export_mrl_onnx.py",
                "--model-id",
                {args.model_id!r},
                "--output-dir",
                "models/mrl",
                "--exporter",
                {args.exporter!r},
            ], cwd=BACKEND)
            run([
                sys.executable,
                "scripts/download_models.py",
                "--profile",
                "production",
                "--quantize",
            ], cwd=BACKEND)
            run([
                sys.executable,
                "scripts/train_projection.py",
                "--teacher",
                "mrl",
                "--mrl-model",
                "models/mrl/model.onnx",
                "--out",
                "models/projection.npz",
                "--dim",
                str({args.dim}),
                "--hash-features",
                str({args.hash_features}),
                "--synthetic-count",
                str({args.synthetic_count}),
                "--max-examples",
                str({args.max_examples}),
            ], cwd=BACKEND)
            run([
                sys.executable,
                "scripts/write_artifact_manifest.py",
                "--output",
                "models/artifacts.json",
            ], cwd=BACKEND)
            archive = ROOT / "waver-artifacts.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(BACKEND / "models", arcname="models")
            shutil.rmtree(BACKEND, ignore_errors=True)
            print(json.dumps({{"artifact_archive": str(archive)}}, indent=2), flush=True)
            return 0

        if __name__ == "__main__":
            raise SystemExit(main())
        """
    ).strip()


def _prepare_kernel_workspace(args: argparse.Namespace, username: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "backend"
    workdir = Path(tempfile.mkdtemp(prefix="waver-kaggle-"))
    del backend_root
    metadata = {
        "id": f"{username}/{args.kernel_slug}",
        "title": "Waver Artifact Build",
        "code_file": "artifact_job.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{username}/{args.source_dataset_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (workdir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (workdir / "artifact_job.py").write_text(_job_source(args))
    return workdir


def _run(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, capture_output=True, env=env)


def _wait_for_kernel(kernel_id: str, timeout_seconds: int, *, env: dict[str, str]) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_status = ""
    while time.monotonic() < deadline:
        result = _run(["kaggle", "kernels", "status", kernel_id], env=env)
        last_status = result.stdout.strip() or result.stderr.strip()
        lowered = last_status.lower()
        print(last_status, flush=True)
        if "complete" in lowered:
            return last_status
        if any(marker in lowered for marker in ("error", "failed", "cancelled")):
            raise RuntimeError(last_status)
        time.sleep(60)
    raise TimeoutError(f"Timed out waiting for {kernel_id}. Last status: {last_status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit Waver artifact training/export to Kaggle and download outputs."
    )
    parser.add_argument("--kernel-slug", default="waver-artifact-build")
    parser.add_argument("--source-dataset-slug", default=DEFAULT_SOURCE_DATASET_SLUG)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--exporter", choices=["auto", "optimum", "torch"], default="auto")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hash-features", type=int, default=262144)
    parser.add_argument("--synthetic-count", type=int, default=800)
    parser.add_argument("--max-examples", type=int, default=3000)
    parser.add_argument("--timeout-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--output-dir", type=Path, default=Path("models/kaggle-output"))
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("kaggle") is None:
        print(
            "The Kaggle CLI is required. Install it with "
            "`uv tool install kaggle` or `python -m pip install kaggle`.",
            file=sys.stderr,
        )
        return 2
    env = _kaggle_env()
    username = _kaggle_username(env)
    _sync_source_dataset(args, username, env)
    workdir = _prepare_kernel_workspace(args, username)
    kernel_id = f"{username}/{args.kernel_slug}"
    try:
        _run(["kaggle", "kernels", "push", "-p", str(workdir)], env=env)
        if args.no_wait:
            print(f"Submitted {kernel_id}. Workspace kept at {workdir}")
            return 0
        _wait_for_kernel(kernel_id, args.timeout_seconds, env=env)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "kernels", "output", kernel_id, "-p", str(args.output_dir), "-o"], env=env)
        print(f"Downloaded Kaggle outputs to {args.output_dir.resolve()}")
        return 0
    finally:
        if args.keep_workdir or args.no_wait:
            print(f"Kaggle workspace: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
