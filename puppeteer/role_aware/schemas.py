from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"

CAPABILITY_DIMENSIONS = (
    "planning",
    "general_reasoning",
    "quantitative_reasoning",
    "domain_reasoning",
    "software_engineering",
    "commonsense_generation",
    "verification",
    "repair",
    "integration",
    "tool_use",
)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _bounded(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1], got {value!r}")
    return value


@dataclass(frozen=True)
class IOSchema:
    type: str
    fields: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "IOSchema":
        data = data or {}
        return cls(
            type=str(data.get("type", "unspecified")),
            fields=_as_tuple(data.get("fields")),
            requirements=_as_tuple(data.get("requirements")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float = 0.1
    max_tokens: int = 4096

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DecodingConfig":
        data = data or {}
        return cls(
            temperature=float(data.get("temperature", 0.1)),
            max_tokens=int(data.get("max_tokens", 4096)),
        )


@dataclass(frozen=True)
class RoleCard:
    role_name: str
    role_goal: str
    core_functions: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    expected_input: IOSchema
    expected_output: IOSchema
    tools: tuple[str, ...]
    capability_prior: Mapping[str, float]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.role_name.strip():
            raise ValueError("RoleCard.role_name must not be empty")
        if not self.role_goal.strip():
            raise ValueError("RoleCard.role_goal must not be empty")
        overlap = set(self.allowed_actions) & set(self.forbidden_actions)
        if overlap:
            raise ValueError(f"Actions cannot be both allowed and forbidden: {sorted(overlap)}")
        unknown = set(self.capability_prior) - set(CAPABILITY_DIMENSIONS)
        if unknown:
            raise ValueError(f"Unknown capability dimensions: {sorted(unknown)}")
        for name, value in self.capability_prior.items():
            _bounded(value, f"capability_prior[{name!r}]")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoleCard":
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            role_name=str(data["role_name"]),
            role_goal=str(data["role_goal"]),
            core_functions=_as_tuple(data.get("core_functions")),
            allowed_actions=_as_tuple(data.get("allowed_actions")),
            forbidden_actions=_as_tuple(data.get("forbidden_actions")),
            expected_input=IOSchema.from_dict(data.get("expected_input")),
            expected_output=IOSchema.from_dict(data.get("expected_output")),
            tools=_as_tuple(data.get("tools")),
            capability_prior={
                str(name): _bounded(value, f"capability_prior[{name!r}]")
                for name, value in (data.get("capability_prior") or {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role_name": self.role_name,
            "role_goal": self.role_goal,
            "core_functions": list(self.core_functions),
            "allowed_actions": list(self.allowed_actions),
            "forbidden_actions": list(self.forbidden_actions),
            "expected_input": self.expected_input.to_dict(),
            "expected_output": self.expected_output.to_dict(),
            "tools": list(self.tools),
            "capability_prior": dict(self.capability_prior),
        }

    def to_prompt(self) -> str:
        parts = [
            f"Role: {self.role_name}",
            f"Goal: {self.role_goal}",
            "Core functions: " + ", ".join(self.core_functions or ("unspecified",)),
            "Allowed actions: " + ", ".join(self.allowed_actions or ("none",)),
            "Forbidden actions: " + ", ".join(self.forbidden_actions or ("none",)),
            "Expected input type: " + self.expected_input.type,
            "Expected output type: " + self.expected_output.type,
            "Tools: " + ", ".join(self.tools or ("none",)),
        ]
        return "\n".join(parts)


@dataclass(frozen=True)
class TeammateSpec:
    teammate_id: str
    backbone: str
    role_card: RoleCard
    provider_profile: str | None = None
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    available: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.teammate_id.strip():
            raise ValueError("TeammateSpec.teammate_id must not be empty")
        if not self.backbone.strip():
            raise ValueError("TeammateSpec.backbone must not be empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TeammateSpec":
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            teammate_id=str(data["teammate_id"]),
            backbone=str(data["backbone"]),
            provider_profile=data.get("provider_profile"),
            decoding=DecodingConfig.from_dict(data.get("decoding")),
            available=bool(data.get("available", True)),
            metadata=dict(data.get("metadata") or {}),
            role_card=RoleCard.from_dict(data["role_card"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "teammate_id": self.teammate_id,
            "backbone": self.backbone,
            "provider_profile": self.provider_profile,
            "decoding": asdict(self.decoding),
            "available": self.available,
            "metadata": dict(self.metadata),
            "role_card": self.role_card.to_dict(),
        }


@dataclass(frozen=True)
class PublicAgentView:
    role_name: str
    role_goal: str
    core_functions: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    expected_input_type: str
    expected_output_type: str
    tools: tuple[str, ...]
    capability_names: tuple[str, ...]
    capability_mean: tuple[float, ...]
    uncertainty: tuple[float, ...]
    observation_count: tuple[int, ...]
    role_adherence: float
    role_adherence_uncertainty: float
    global_reliability: float
    global_reliability_uncertainty: float
    global_reliability_count: int
    available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityProfile:
    capability_names: tuple[str, ...]
    mean: list[float]
    uncertainty: list[float]
    observation_count: list[int]
    role_adherence_mean: float = 1.0
    role_adherence_uncertainty: float = 1.0
    role_adherence_count: int = 0
    global_reliability_mean: float = 0.5
    global_reliability_uncertainty: float = 1.0
    global_reliability_count: int = 0
    update_step: int = 0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        size = len(self.capability_names)
        if not (len(self.mean) == len(self.uncertainty) == len(self.observation_count) == size):
            raise ValueError("CapabilityProfile vector lengths must match capability_names")
        if len(set(self.capability_names)) != size:
            raise ValueError("CapabilityProfile capability_names must be unique")
        self.mean = [_bounded(value, "mean") for value in self.mean]
        self.uncertainty = [_bounded(value, "uncertainty") for value in self.uncertainty]
        self.role_adherence_mean = _bounded(self.role_adherence_mean, "role_adherence_mean")
        self.role_adherence_uncertainty = _bounded(
            self.role_adherence_uncertainty, "role_adherence_uncertainty"
        )
        self.global_reliability_mean = _bounded(
            self.global_reliability_mean, "global_reliability_mean"
        )
        self.global_reliability_uncertainty = _bounded(
            self.global_reliability_uncertainty, "global_reliability_uncertainty"
        )

    @classmethod
    def from_role_card(cls, role_card: RoleCard) -> "CapabilityProfile":
        mean = [float(role_card.capability_prior.get(name, 0.5)) for name in CAPABILITY_DIMENSIONS]
        return cls(
            capability_names=CAPABILITY_DIMENSIONS,
            mean=mean,
            uncertainty=[1.0] * len(CAPABILITY_DIMENSIONS),
            observation_count=[0] * len(CAPABILITY_DIMENSIONS),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityProfile":
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            capability_names=tuple(data["capability_names"]),
            mean=list(data["mean"]),
            uncertainty=list(data["uncertainty"]),
            observation_count=list(data["observation_count"]),
            role_adherence_mean=float(data.get("role_adherence_mean", 1.0)),
            role_adherence_uncertainty=float(data.get("role_adherence_uncertainty", 1.0)),
            role_adherence_count=int(data.get("role_adherence_count", 0)),
            global_reliability_mean=float(data.get("global_reliability_mean", 0.5)),
            global_reliability_uncertainty=float(
                data.get("global_reliability_uncertainty", 1.0)
            ),
            global_reliability_count=int(data.get("global_reliability_count", 0)),
            update_step=int(data.get("update_step", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update_capability(self, capability: str, reward: float, alpha: float) -> None:
        if capability not in self.capability_names:
            raise KeyError(f"Unknown capability {capability!r}")
        alpha = _bounded(alpha, "alpha")
        reward = _bounded(reward, "reward")
        index = self.capability_names.index(capability)
        self.mean[index] = (1.0 - alpha) * self.mean[index] + alpha * reward
        self.observation_count[index] += 1
        self.uncertainty[index] = 1.0 / math.sqrt(self.observation_count[index] + 1.0)
        self.update_step += 1

    def update_capability_batch(
        self,
        capability: str,
        mean_reward: float,
        alpha: float,
        observation_count: int,
    ) -> None:
        if capability not in self.capability_names:
            raise KeyError(f"Unknown capability {capability!r}")
        if observation_count <= 0:
            raise ValueError("observation_count must be positive")
        alpha = _bounded(alpha, "alpha")
        mean_reward = _bounded(mean_reward, "mean_reward")
        alpha_batch = 1.0 - (1.0 - alpha) ** observation_count
        index = self.capability_names.index(capability)
        self.mean[index] = (
            (1.0 - alpha_batch) * self.mean[index] + alpha_batch * mean_reward
        )
        self.observation_count[index] += observation_count
        self.uncertainty[index] = 1.0 / math.sqrt(
            self.observation_count[index] + 1.0
        )
        self.update_step += observation_count

    def update_global_reliability_batch(
        self, mean_reward: float, alpha: float, observation_count: int
    ) -> None:
        if observation_count <= 0:
            raise ValueError("observation_count must be positive")
        alpha = _bounded(alpha, "alpha")
        mean_reward = _bounded(mean_reward, "mean_reward")
        alpha_batch = 1.0 - (1.0 - alpha) ** observation_count
        self.global_reliability_mean = (
            (1.0 - alpha_batch) * self.global_reliability_mean
            + alpha_batch * mean_reward
        )
        self.global_reliability_count += observation_count
        self.global_reliability_uncertainty = 1.0 / math.sqrt(
            self.global_reliability_count + 1.0
        )
        self.update_step += observation_count

    def update_role_adherence(self, adhered: bool, alpha: float) -> None:
        alpha = _bounded(alpha, "alpha")
        reward = 1.0 if adhered else 0.0
        self.role_adherence_mean = (
            (1.0 - alpha) * self.role_adherence_mean + alpha * reward
        )
        self.role_adherence_count += 1
        self.role_adherence_uncertainty = 1.0 / math.sqrt(self.role_adherence_count + 1.0)
        self.update_step += 1

    def public_view(self, spec: TeammateSpec) -> PublicAgentView:
        card = spec.role_card
        return PublicAgentView(
            role_name=card.role_name,
            role_goal=card.role_goal,
            core_functions=card.core_functions,
            allowed_actions=card.allowed_actions,
            forbidden_actions=card.forbidden_actions,
            expected_input_type=card.expected_input.type,
            expected_output_type=card.expected_output.type,
            tools=card.tools,
            capability_names=self.capability_names,
            capability_mean=tuple(self.mean),
            uncertainty=tuple(self.uncertainty),
            observation_count=tuple(self.observation_count),
            role_adherence=self.role_adherence_mean,
            role_adherence_uncertainty=self.role_adherence_uncertainty,
            global_reliability=self.global_reliability_mean,
            global_reliability_uncertainty=self.global_reliability_uncertainty,
            global_reliability_count=self.global_reliability_count,
            available=spec.available,
        )


@dataclass(frozen=True)
class CapabilityEvidence:
    teammate_id: str
    capability: str
    reward: float
    source: str
    interaction_id: str
    role_adhered: bool | None = None
