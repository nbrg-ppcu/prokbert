from __future__ import annotations

import argparse
from pathlib import Path

from .training_helper import DEFAULT_REGISTRY_REPO_ID, DEFAULT_REGISTRY_REVISION, sync_registry_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync ProkBERT model registry from the HF dataset repo.")
    parser.add_argument("local_dir", help="Local target directory for the registry snapshot.")
    parser.add_argument("--repo-id", default=DEFAULT_REGISTRY_REPO_ID, help="HF dataset repo id.")
    parser.add_argument("--revision", default=DEFAULT_REGISTRY_REVISION, help="HF dataset revision.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    args = parser.parse_args()

    snapshot_path = sync_registry_snapshot(
        local_dir=Path(args.local_dir),
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    print(snapshot_path)


if __name__ == "__main__":
    main()
