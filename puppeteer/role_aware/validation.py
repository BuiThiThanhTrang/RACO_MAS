from __future__ import annotations

import json
import re
from typing import Iterable

from role_aware.schemas import PublicAgentView, RoleCard, TeammateSpec


INTERNAL_FIELD_NAMES = {
    "teammate_id",
    "backbone",
    "provider_profile",
    "hash",
    "model_type",
    "model",
    "index",
}


def validate_role_card_identity(card: RoleCard, forbidden_terms: Iterable[str]) -> None:
    public_text = json.dumps(card.to_dict(), ensure_ascii=False).casefold()
    leaks = sorted(
        term for term in {str(item).strip().casefold() for item in forbidden_terms} if term and term in public_text
    )
    if leaks:
        raise ValueError(f"RoleCard leaks model/provider identity: {leaks}")


def assert_public_view_has_no_internal_fields(view: PublicAgentView) -> None:
    serialized = json.dumps(view.to_dict(), ensure_ascii=False, sort_keys=True)
    keys = set(re.findall(r'"([^"]+)"\s*:', serialized))
    overlap = keys & INTERNAL_FIELD_NAMES
    if overlap:
        raise ValueError(f"PublicAgentView exposes internal fields: {sorted(overlap)}")


def validate_teammate_specs(specs: Iterable[TeammateSpec]) -> tuple[TeammateSpec, ...]:
    ordered = tuple(specs)
    ids = [spec.teammate_id for spec in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("teammate_id values must be unique")
    for spec in ordered:
        validate_role_card_identity(
            spec.role_card,
            forbidden_terms=(spec.teammate_id, spec.backbone, spec.provider_profile or ""),
        )
    return ordered
