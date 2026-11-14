"""Publish the prepared public-safe RoleRouteBench beta.2 folder to the Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


DELETE_PATTERNS = [
    "data/actor_inputs/**",
    "data/executor_outputs/**",
    "data/interactive_evaluator/**",
    "data/recovery_annotations/**",
    "data/track_c_labels/**",
    "data/track_c_pairs/**",
    "data/track_c_raw_development/**",
    "data/track_c_runtime/**",
    "data/track_c_state/**",
    "data/track_c_structured/**",
    "specs/post_fault_input.schema.json",
    "specs/post_fault_label.schema.json",
    "specs/post_fault_pair.schema.json",
    "specs/post_fault_runtime.schema.json",
    "specs/recovery.schema.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-id", default="TrangBui/RoleRouteBench")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--parent-commit", required=True)
    parser.add_argument("--include-private", action="store_true")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if folder.name != "RoutingBench-V4" or not (folder / "track_b_package_report.json").is_file():
        raise ValueError(f"Refusing to publish unexpected folder: {folder}")
    has_private = any(folder.glob("track_b_packets/*/*/private"))
    if has_private and not args.include_private:
        raise ValueError("Private packet directories require explicit --include-private approval")

    api = HfApi()
    identity = api.whoami()
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=folder,
        revision=args.revision,
        parent_commit=args.parent_commit,
        commit_message="Publish V4 beta.2 live Track B domains",
        commit_description=(
            "Keep Track A unchanged; publish four complete live Track B domains; "
            "temporarily remove Track C. Evaluator-only private artifacts are public "
            "by explicit project-owner approval and remain excluded from router input."
        ),
        ignore_patterns=[".git/**", ".git"],
        delete_patterns=DELETE_PATTERNS,
    )
    print(
        json.dumps(
            {
                "authenticated_user": identity.get("name"),
                "repo_id": args.repo_id,
                "revision": args.revision,
                "commit_oid": result.oid,
                "commit_url": result.commit_url,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
