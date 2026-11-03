from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


STRUCTURED_OPERATIONS = (
    "normalize_sales",
    "join_declared_key",
    "convert_timezone",
    "aggregate_by_group",
    "deduplicate_precedence",
    "pivot_categories",
    "impute_permitted_missing",
    "filter_and_reconcile",
    "flatten_nested_records",
    "rolling_statistics",
    "map_controlled_vocabulary",
    "merge_latest_versions",
    "reshape_survey",
)


def structured_transform(
    rows: Sequence[Mapping[str, Any]], operation: str
) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    if operation == "normalize_sales":
        return [
            {
                **row,
                "category": str(row["category"]).strip().lower(),
                "net_total": row["quantity"] * row["unit_price"] - row["discount"],
            }
            for row in values
        ]
    if operation == "join_declared_key":
        labels = {"A": "alpha", "B": "beta"}
        return [{**row, "group_label": labels.get(row["group"], "unmatched")} for row in values]
    if operation == "convert_timezone":
        return [{**row, "local_timestamp": str(row["timestamp"]).replace("Z", "+07:00")} for row in values]
    if operation == "aggregate_by_group":
        totals: dict[str, int] = defaultdict(int)
        for row in values:
            totals[str(row["group"])] += int(row["quantity"])
        return [{"group": key, "quantity_sum": totals[key]} for key in sorted(totals)]
    if operation == "deduplicate_precedence":
        latest: dict[str, dict[str, Any]] = {}
        for row in values:
            key = str(row["entity_id"])
            if key not in latest or int(row["version"]) > int(latest[key]["version"]):
                latest[key] = row
        return [latest[key] for key in sorted(latest)]
    if operation == "pivot_categories":
        counts: dict[str, int] = defaultdict(int)
        for row in values:
            counts[str(row["category"]).strip().lower()] += 1
        return [{"metric": "count", **{key: counts[key] for key in sorted(counts)}}]
    if operation == "impute_permitted_missing":
        return [{**row, "survey_score": 3 if row.get("survey_score") is None else row["survey_score"]} for row in values]
    if operation == "filter_and_reconcile":
        retained = [row for row in values if row["quantity"] * row["unit_price"] >= 20]
        return [{**row, "gross_total": row["quantity"] * row["unit_price"]} for row in retained]
    if operation == "flatten_nested_records":
        return [{**{key: value for key, value in row.items() if key != "nested"}, **dict(row.get("nested") or {})} for row in values]
    if operation == "rolling_statistics":
        total = 0
        result = []
        for row in values:
            total += int(row["quantity"])
            result.append({**row, "rolling_quantity_sum": total})
        return result
    if operation == "map_controlled_vocabulary":
        mapping = {" Retail ": "retail", "WHOLESALE": "wholesale", "Unknown": "unmapped"}
        return [{**row, "controlled_category": mapping.get(str(row["category"]), "unmapped")} for row in values]
    if operation == "merge_latest_versions":
        return structured_transform(values, "deduplicate_precedence")
    if operation == "reshape_survey":
        return [{"respondent_id": row["record_id"], "score": row.get("survey_score"), "segment": str(row["category"]).strip().lower()} for row in values]
    raise ValueError(f"Unsupported structured operation: {operation}")


SCIENTIFIC_OPERATIONS = (
    "calibrate_controls",
    "normalize_baseline",
    "derive_concentration",
    "replicate_quality_check",
    "apply_correction_factor",
    "compare_conditions",
    "derive_rate",
    "threshold_classification",
    "estimate_parameter",
    "dilution_monotonicity",
    "reconcile_control",
    "material_balance",
    "apply_exclusion_rule",
)


def scientific_result(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(protocol["operation"])
    values = [float(row["measurement"]) for row in rows]
    control = float(protocol["control"])
    if operation == "calibrate_controls":
        result: Any = round(sum(value - control for value in values) / len(values), 3)
    elif operation == "normalize_baseline":
        result = round(sum(value / control for value in values) / len(values), 3)
    elif operation == "derive_concentration":
        result = round(sum(values) / len(values) * float(protocol["factor"]), 3)
    elif operation == "replicate_quality_check":
        result = round(max(values) - min(values), 3)
    elif operation == "apply_correction_factor":
        result = round(sum(values) / len(values) * float(protocol["factor"]), 3)
    elif operation == "compare_conditions":
        midpoint = len(values) // 2
        result = round(sum(values[midpoint:]) / len(values[midpoint:]) - sum(values[:midpoint]) / len(values[:midpoint]), 3)
    elif operation == "derive_rate":
        result = round((values[-1] - values[0]) / (len(values) - 1), 3)
    elif operation == "threshold_classification":
        result = sum(value >= float(protocol["threshold"]) for value in values)
    elif operation == "estimate_parameter":
        result = round(sum(values) / len(values), 3)
    elif operation == "dilution_monotonicity":
        result = all(left >= right for left, right in zip(values, values[1:]))
    elif operation == "reconcile_control":
        result = round(max(abs(value - control) for value in values), 3)
    elif operation == "material_balance":
        result = round(sum(values), 3)
    elif operation == "apply_exclusion_rule":
        retained = [value for value in values if value <= float(protocol["exclusion_max"])]
        result = round(sum(retained) / len(retained), 3)
    else:
        raise ValueError(f"Unsupported scientific operation: {operation}")
    return {"operation": operation, "result": result, "unit": protocol["unit"], "sample_count": len(values)}
