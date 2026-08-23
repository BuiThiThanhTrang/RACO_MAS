from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from role_aware.pool_factory import build_pool

OUTPUT_DIR = Path("personas/role_aware")

def write_pool(name: str, source_scenario: str, count: int = 2) -> None:
    specs = [
        replace(
            spec,
            teammate_id=spec.teammate_id.replace(source_scenario, name, 1),
            metadata=dict(spec.metadata) | {"scenario": name},
        )
        for spec in build_pool(source_scenario, teammates_per_role=count)
    ]
    (OUTPUT_DIR / f"{name}_pool.jsonl").write_text(
        "".join(json.dumps(spec.to_dict(), ensure_ascii=False) + "\n" for spec in specs),
        encoding="utf-8",
    )

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in ("s0", "s1", "s2_backbone", "s2_role", "s2_tools", "s3", "s3_mixed"):
        write_pool(scenario, scenario)
    write_pool("s2_cardinality_small", "s2_cardinality", count=1)
    write_pool("s2_cardinality_large", "s2_cardinality", count=3)

if __name__ == "__main__":
    main()
