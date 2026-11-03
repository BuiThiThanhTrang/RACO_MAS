from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import wave
import zipfile

from .schema import (
    ArtifactManifest,
    sha256_file,
    validate_manifest_shape,
)


def _validate_openable(path: Path, media_type: str) -> str | None:
    try:
        if media_type == "application/json":
            json.loads(path.read_text(encoding="utf-8"))
        elif media_type == "text/csv":
            import csv

            with path.open(encoding="utf-8", newline="") as handle:
                list(csv.DictReader(handle))
        elif media_type.startswith("image/"):
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
        elif media_type == "application/pdf":
            try:
                from pdfminer.pdfpage import PDFPage

                with path.open("rb") as handle:
                    if not list(PDFPage.get_pages(handle)):
                        raise ValueError("PDF contains no pages")
            except ModuleNotFoundError:
                import pymupdf

                with pymupdf.open(path) as document:
                    if document.page_count <= 0:
                        raise ValueError("PDF contains no pages")
        elif media_type == "audio/wav":
            with wave.open(str(path), "rb") as handle:
                if handle.getnframes() <= 0:
                    raise ValueError("WAV contains no frames")
        elif media_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            from pptx import Presentation

            presentation = Presentation(path)
            if not presentation.slides:
                raise ValueError("PPTX contains no slides")
        elif media_type == "application/zip":
            with zipfile.ZipFile(path, "r") as archive:
                if archive.testzip() is not None or not archive.namelist():
                    raise ValueError("ZIP container is empty or corrupted")
    except Exception as error:
        return f"unopenable:{path.name}:{type(error).__name__}:{error}"
    return None


def load_manifest(path: str | Path) -> ArtifactManifest:
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    manifest = ArtifactManifest.from_dict(value)
    validate_manifest_shape(manifest)
    return manifest


def validate_artifact_pack(
    pack_dir: str | Path,
    *,
    include_private: bool = True,
) -> dict[str, Any]:
    pack_dir = Path(pack_dir).resolve()
    manifest = load_manifest(pack_dir)
    errors: list[str] = []
    checked = [*manifest.public_files, *manifest.fixture_files]
    if include_private:
        checked.extend(manifest.private_files)
    for item in checked:
        path = (pack_dir / item.path).resolve()
        try:
            path.relative_to(pack_dir)
        except ValueError:
            errors.append(f"path_escape:{item.path}")
            continue
        if not path.is_file():
            errors.append(f"missing_file:{item.path}")
            continue
        actual = sha256_file(path)
        if actual != item.sha256:
            errors.append(f"checksum_mismatch:{item.path}")
        open_error = _validate_openable(path, item.media_type)
        if open_error:
            errors.append(f"{item.path}:{open_error}")
    expected_paths = {
        "manifest.json",
        *(item.path for item in manifest.public_files),
        *(item.path for item in manifest.private_files),
        *(item.path for item in manifest.fixture_files),
    }
    actual_paths = {
        path.relative_to(pack_dir).as_posix()
        for path in pack_dir.rglob("*")
        if path.is_file()
    }
    for path in sorted(actual_paths - expected_paths):
        errors.append(f"unlisted_file:{path}")
    return {
        "valid": not errors,
        "base_workflow_id": manifest.base_workflow_id,
        "domain": manifest.domain,
        "public_file_count": len(manifest.public_files),
        "private_file_count": len(manifest.private_files),
        "fixture_file_count": len(manifest.fixture_files),
        "errors": errors,
    }
