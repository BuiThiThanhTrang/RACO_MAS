from __future__ import annotations

from importlib.resources import files


def load_prompt(name: str) -> str:
    return files("routerlab").joinpath("prompts", f"{name}.txt").read_text(
        encoding="utf-8"
    ).strip()
