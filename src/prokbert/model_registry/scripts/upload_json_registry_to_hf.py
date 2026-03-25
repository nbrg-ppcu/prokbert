#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the JSON registry folder to a HF dataset repository.")
    parser.add_argument("folder", help="Local folder containing README.md and registry JSON files")
    parser.add_argument("--repo-id", default="neuralbioinfo/model-registry", help="Target HF dataset repository id")
    parser.add_argument("--token", default=None, help="Optional HF token")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(folder)

    api = HfApi(token=args.token)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {folder} to dataset repo {args.repo_id}")


if __name__ == "__main__":
    main()
