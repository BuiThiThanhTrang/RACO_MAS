from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ID = "gaia-benchmark/GAIA"


def _patterns(split: str):
    if split == "all":
        return None
    return ["README.md", f"2023/{split}/**"]


def main():
    parser = argparse.ArgumentParser(
        description="Download the gated official GAIA snapshot without printing examples."
    )
    parser.add_argument(
        "--split", choices=["validation", "test", "all"], default="validation"
    )
    parser.add_argument("--output", default="data/GAIA")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, get_token, snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is missing. Install the project requirements first."
        ) from error

    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face authentication was found. Open "
            "https://huggingface.co/datasets/gaia-benchmark/GAIA, accept the "
            "access terms, then either run `hf auth login` or set $env:HF_TOKEN "
            "to a read token in this PowerShell session."
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        info = HfApi().dataset_info(REPO_ID, revision=args.revision, token=token)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=info.sha,
            token=token,
            local_dir=output,
            allow_patterns=_patterns(args.split),
        )
    except GatedRepoError as error:
        raise SystemExit(
            "Your Hugging Face account has not been granted GAIA access. Accept "
            "the dataset terms with the same account that owns HF_TOKEN."
        ) from error
    except HfHubHTTPError as error:
        raise SystemExit(f"GAIA download failed: {error}") from error

    required_splits = ["validation", "test"] if args.split == "all" else [args.split]
    for split in required_splits:
        metadata = output / "2023" / split / "metadata.parquet"
        if not metadata.is_file():
            raise SystemExit(f"Download is incomplete; missing {metadata}")

    manifest = {
        "repo_id": REPO_ID,
        "revision": info.sha,
        "downloaded_split": args.split,
        "local_dir": str(output.resolve()),
    }
    (output / ".snapshot.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"GAIA {args.split} downloaded to {output.resolve()} at revision {info.sha}."
    )


if __name__ == "__main__":
    main()
