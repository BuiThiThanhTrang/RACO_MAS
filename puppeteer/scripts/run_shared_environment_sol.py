"""Run the frozen shared Track B environment with Sol as the router.

This wrapper leaves ``run_shared_environment.py`` unchanged.  It only adds the
already-existing ``StructuredLLMRouter`` adapter and includes itself in the
per-run source hash manifest.
"""
from __future__ import annotations

from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from routing_benchmark.routers import StructuredLLMRouter
from routing_benchmark.v4.artifacts.schema import sha256_file
from scripts import run_shared_environment as shared


def build_router(configuration, seed):
    config = dict(configuration)
    kind = config.get("type", "decision")
    if kind != "llm":
        return shared._sol_original_build_router(configuration, seed)
    profile = str(config.get("orchestration_profile", "structured"))
    guard = config.get("observable_rejection_guard_enabled", False)
    if not isinstance(guard, bool):
        raise ValueError("observable_rejection_guard_enabled must be a boolean")
    return StructuredLLMRouter(
        model=str(config["model"]),
        orchestration_profile=profile,
        reasoning_effort=str(config.get("reasoning_effort", "low")),
        max_repair_attempts=int(config.get("max_repair_attempts", 1)),
        observable_rejection_guard_enabled=guard,
    )


def source_hashes(packet_root):
    digest = shared._sol_original_source_hashes(packet_root)
    digest[str(Path(__file__).resolve().relative_to(PROJECT))] = sha256_file(
        Path(__file__).resolve()
    )
    return digest


shared._sol_original_build_router = shared.build_router
shared._sol_original_source_hashes = shared.source_hashes
shared.build_router = build_router
shared.source_hashes = source_hashes


if __name__ == "__main__":
    shared.main()
