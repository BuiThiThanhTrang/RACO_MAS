from __future__ import annotations

import json
from pathlib import Path

from model.model_config import model_registry
from role_aware.schemas import TeammateSpec
from role_aware.validation import validate_teammate_specs


def load_teammate_specs(path: str | Path) -> tuple[TeammateSpec, ...]:
    source = Path(path)
    specs: list[TeammateSpec] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            spec = TeammateSpec.from_dict(raw)
        except Exception as exc:
            raise ValueError(f"Invalid teammate persona at {source}:{line_number}: {exc}") from exc
        if model_registry.get_model_config(spec.backbone) is None:
            raise ValueError(
                f"Unknown backbone {spec.backbone!r} at {source}:{line_number}. "
                "Register it in model/model_config.py before using this persona."
            )
        specs.append(spec)
    if not specs:
        raise ValueError(f"No teammate personas found in {source}")
    return validate_teammate_specs(specs)
