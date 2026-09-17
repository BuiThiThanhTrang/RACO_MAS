"""Run on a CPU Modal Function before provisioning inference GPUs."""
import hashlib
from pathlib import Path
from .artifacts import LOCK, atomic_json, model_path, tokenizer_path


def download(root, include_reference=False):
    from huggingface_hub import hf_hub_download, snapshot_download
    root = Path(root)
    target = model_path(root)
    hf_hub_download(LOCK["gguf_repo"], LOCK["gguf_filename"],
                    revision=LOCK["gguf_revision"], local_dir=target.parent)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    if target.stat().st_size != LOCK["gguf_bytes"] or digest.hexdigest() != LOCK["gguf_sha256"]:
        raise RuntimeError("GGUF checksum/size mismatch; do not serve this artifact")
    snapshot_download(LOCK["model_id"], revision=LOCK["model_revision"],
        local_dir=tokenizer_path(root),
        allow_patterns=["tokenizer*", "special_tokens_map.json", "config.json", "generation_config.json"])
    if include_reference:
        snapshot_download(LOCK["model_id"], revision=LOCK["model_revision"],
            local_dir=root / "reference" / LOCK["model_revision"],
            allow_patterns=["*.safetensors", "*.json", "tokenizer.model"])
    atomic_json(root / "download.json", LOCK)
