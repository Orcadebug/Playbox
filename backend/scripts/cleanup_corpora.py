from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal
from app.services.corpora import CorpusService


async def _run(*, delete_expired: bool) -> dict[str, int]:
    async with SessionLocal() as session:
        service = CorpusService(session)
        expired = await service.expire_due_corpora()
        deleted = await service.delete_expired_corpora() if delete_expired else 0
    return {"expired": expired, "deleted": deleted}


def main() -> int:
    parser = argparse.ArgumentParser(description="Expire and optionally delete session corpora.")
    parser.add_argument(
        "--delete-expired",
        action="store_true",
        help="Physically delete corpora already marked expired after expiring due corpora.",
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(delete_expired=args.delete_expired)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
