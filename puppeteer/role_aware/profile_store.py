from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from role_aware.schemas import (
    CapabilityEvidence,
    CapabilityProfile,
    PublicAgentView,
    TeammateSpec,
)


class ProfileStore:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = float(alpha)
        self._profiles: dict[str, CapabilityProfile] = {}
        self._priors: dict[str, CapabilityProfile] = {}

    def initialize(self, specs: Iterable[TeammateSpec]) -> None:
        self._profiles.clear()
        self._priors.clear()
        for spec in specs:
            if spec.teammate_id in self._profiles:
                raise ValueError(f"Duplicate teammate_id {spec.teammate_id!r}")
            profile = CapabilityProfile.from_role_card(spec.role_card)
            self._profiles[spec.teammate_id] = profile
            self._priors[spec.teammate_id] = CapabilityProfile.from_dict(profile.to_dict())

    def get(self, teammate_id: str) -> CapabilityProfile:
        try:
            return self._profiles[teammate_id]
        except KeyError as exc:
            raise KeyError(f"No capability profile for teammate {teammate_id!r}") from exc

    def update(self, evidence: CapabilityEvidence) -> None:
        profile = self.get(evidence.teammate_id)
        profile.update_capability(evidence.capability, evidence.reward, self.alpha)
        if evidence.role_adhered is not None:
            profile.update_role_adherence(evidence.role_adhered, self.alpha)

    def update_terminal_batch(
        self,
        teammate_id: str,
        capabilities: Iterable[str],
        mean_reward: float,
        observation_count: int,
        role_adhered: bool | None = None,
    ) -> None:
        profile = self.get(teammate_id)
        profile.update_global_reliability_batch(
            mean_reward, self.alpha, observation_count
        )
        for capability in tuple(dict.fromkeys(capabilities)):
            profile.update_capability_batch(
                capability, mean_reward, self.alpha, observation_count
            )
        if role_adhered is not None:
            profile.update_role_adherence(role_adhered, self.alpha)

    def reset(self, teammate_ids: Iterable[str] | None = None) -> None:
        ids = tuple(teammate_ids) if teammate_ids is not None else tuple(self._profiles)
        for teammate_id in ids:
            self._profiles[teammate_id] = CapabilityProfile.from_dict(
                self._priors[teammate_id].to_dict()
            )

    def public_views(self, specs: Iterable[TeammateSpec]) -> tuple[PublicAgentView, ...]:
        return tuple(self.get(spec.teammate_id).public_view(spec) for spec in specs)

    def to_dict(self) -> dict[str, dict]:
        return {teammate_id: profile.to_dict() for teammate_id, profile in self._profiles.items()}

    def state_dict(self) -> dict[str, dict]:
        return self.to_dict()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        try:
            with temporary.open("w", encoding="utf-8") as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def load_file(self, path: str | Path, strict: bool = True) -> None:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Capability profile file not found: {target}")
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Capability profile file must contain an object")
        self.load_state_dict(data, strict=strict)

    def load_state_dict(
        self, data: Mapping[str, Mapping], strict: bool = True
    ) -> None:
        expected = set(self._profiles)
        actual = set(data)
        if strict and actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"Capability profile teammate mismatch; missing={missing}, extra={extra}"
            )
        for teammate_id, profile_data in data.items():
            if teammate_id not in self._profiles:
                raise KeyError(f"Cannot load profile for unknown teammate {teammate_id!r}")
            self._profiles[teammate_id] = CapabilityProfile.from_dict(profile_data)

    def load(self, data: Mapping[str, Mapping]) -> None:
        self.load_state_dict(data, strict=False)
