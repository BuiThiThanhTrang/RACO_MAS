"""Pinned artifacts and verification identity."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
LOCK = json.loads((PACKAGE / "model.lock.json").read_text())


def fingerprint():
    digest = hashlib.sha256()
    for name in ("model.lock.json", "backend.py", "schemas.py", "probe.py", "modal_app.py", "artifacts.py", "download_model.py", "probe_cases.json"):
        digest.update(name.encode())
        digest.update((PACKAGE / name).read_bytes())
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def model_path(root):
    return Path(root) / "gguf" / LOCK["gguf_revision"] / LOCK["gguf_filename"]


def tokenizer_path(root):
    return Path(root) / "tokenizer" / LOCK["model_revision"]


def require_download(root):
    root = Path(root)
    command = "python -m modal run -m deploy.reward_model.modal_app::download"
    try:
        manifest = json.loads((root / "download.json").read_text())
        size = model_path(root).stat().st_size
    except (FileNotFoundError, json.JSONDecodeError):
        raise RuntimeError(
            f"Model download is missing or incomplete in the mounted Volume at {root}. "
            f"Run `{command}` and wait for checksum verification to finish before probe_q5. "
            "Use --include-reference as well before probe_reference."
        ) from None
    if manifest != LOCK or size != LOCK["gguf_bytes"]:
        raise RuntimeError(f"Pinned artifacts are missing or changed; rerun `{command}`")


def require_verified(root):
    data = json.loads((Path(root) / "verification.json").read_text())
    if data.get("fingerprint") != fingerprint() or data.get("passed") is not True:
        raise RuntimeError("Q5 backend has not passed verification for this code/model revision")
