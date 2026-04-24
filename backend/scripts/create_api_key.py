from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.auth import generate_api_key, hash_api_key
from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models import ApiKey


async def _create(workspace: str, name: str | None) -> str:
    settings = get_settings()
    token = generate_api_key()
    await init_db()
    async with SessionLocal() as session:
        row = ApiKey(
            workspace_id=workspace,
            name=name,
            key_hash=hash_api_key(token),
            requests_per_minute=settings.waver_default_requests_per_minute,
            bytes_per_minute=settings.waver_default_bytes_per_minute,
        )
        session.add(row)
        await session.commit()
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Waver beta API key.")
    parser.add_argument("--workspace", required=True, help="Workspace id bound to this key")
    parser.add_argument("--name", default=None, help="Optional display name")
    args = parser.parse_args()
    token = asyncio.run(_create(args.workspace, args.name))
    print(token)
    print("Store this token now; only its SHA-256 hash was saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
