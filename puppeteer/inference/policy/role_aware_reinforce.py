from __future__ import annotations

import copy
import datetime
import json
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn

from inference.policy.base_policy import LearningPolicy
from role_aware.capability_scopes import ROLE_CAPABILITY_SCOPES
from role_aware.reward_scorer import AuxiliaryRewardScorer
from role_aware.schemas import CAPABILITY_DIMENSIONS
from role_aware.task_analyzer import TaskAnalyzerRepresentation
from role_aware.trajectory_reward import build_trajectory_reward_input
from role_aware.multiselect import sample_ordered

logger = logging.getLogger("train")
ORCHESTRATOR_STOP = "__orchestrator_stop__"
ROLE_NAMES = tuple(ROLE_CAPABILITY_SCOPES)


class CandidateScoringPolicyNetwork(nn.Module):
    """Task analyzer + agent encoder + Set Transformer + shared scorer."""

    def __init__(self, state_dim: int, candidate_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.agent_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.set_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        self.set_norm_1 = nn.LayerNorm(hidden_dim)
        self.set_feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.set_norm_2 = nn.LayerNorm(hidden_dim)
        self.shared_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.stop_embedding = nn.Parameter(torch.zeros(candidate_dim))

    def forward(self, state, candidate_features, include_stop=True):
        state = state.to(torch.float32)
        candidate_features = candidate_features.to(torch.float32)
        if include_stop:
            candidate_features = torch.cat(
                (candidate_features, self.stop_embedding.unsqueeze(0)), dim=0
            )

        task_vector = self.state_encoder(state)
        encoded_agents = self.agent_encoder(candidate_features)
        encoded_agents = encoded_agents.unsqueeze(0).expand(state.shape[0], -1, -1)
        attended, _ = self.set_attention(
            encoded_agents, encoded_agents, encoded_agents, need_weights=False
        )
        contextual = self.set_norm_1(encoded_agents + attended)
        contextual = self.set_norm_2(
            contextual + self.set_feed_forward(contextual)
        )
        expanded_task = task_vector.unsqueeze(1).expand(-1, contextual.shape[1], -1)
        logits = self.shared_scorer(
            torch.cat((expanded_task, contextual), dim=-1)
        ).squeeze(-1)
        return torch.softmax(logits, dim=-1)


class RoleAwareREINFORCE(LearningPolicy):
    def __init__(self, agent_graph, action_graph, config, runtime_config):
        super().__init__(agent_graph, action_graph)
        if config is None:
            raise ValueError("RoleAwareREINFORCE requires an explicit per-run config")
        self.config = copy.deepcopy(dict(config))
        self.runtime_config = copy.deepcopy(dict(runtime_config))
        self.device = self.config.get("device", {}).get("type", "cpu")
        self.dataset_mode = str(self.config.get("dataset_mode", "train"))
        self.policy_mode = str(self.config.get("policy_mode", "train"))
        training = self.config.get("training", {})
        requested_training = bool(training.get("training", True))
        self.optimizer_updates_enabled = bool(
            requested_training
            and self.dataset_mode == "train"
            and self.policy_mode == "train"
        )
        self.training = self.optimizer_updates_enabled
        self.loading = bool(training.get("loading", False))
        self.learning_rate = float(training.get("learning_rate", 1e-4))
        self.gamma = float(training.get("gamma", 0.99))
        self.entropy_coef = float(training.get("entropy_coef", 0.01))
        reward_mode_config = dict(self.config.get("reward", {}) or {})
        self.reward_mode = str(reward_mode_config.get("mode", "role_aware_v1"))
        if self.reward_mode not in {"role_aware_v1", "baseline_compatible_v1"}:
            raise ValueError(
                "policy.reward.mode must be role_aware_v1 or baseline_compatible_v1"
            )
        self.baseline_reward_config = dict(
            reward_mode_config.get("baseline_compatible", {}) or {}
        )
        self.threshold_margin_coef = float(training.get("threshold_margin_coef", 0.0))
        self.threshold_margin_cap = float(training.get("threshold_margin_cap", 0.25))
        if self.threshold_margin_coef < 0:
            raise ValueError("training.threshold_margin_coef must be nonnegative")
        if self.threshold_margin_cap <= 0:
            raise ValueError("training.threshold_margin_cap must be positive")
        policy_cost = self.config.get("cost", {})
        cost_config = self.runtime_config.get("cost_reward", {})
        self.step_penalty = float(
            cost_config.get(
                "step_penalty",
                policy_cost.get("step_penalty", policy_cost.get("scale", 0.1)),
            )
        )
        self.token_cost_weight = float(
            cost_config.get(
                "token_cost_weight",
                policy_cost.get("token_cost_weight", policy_cost.get("scale", 0.1)),
            )
        )
        self.cost_normalization = float(
            cost_config.get("normalization", 100000.0)
        )
        if self.cost_normalization <= 0:
            raise ValueError("cost_reward.normalization must be positive")
        routing = self.config.get("routing", {})
        self.routing_mode = routing.get("mode", "legacy_threshold")
        if self.routing_mode not in {
            "legacy_threshold",
            "categorical_set_v2",
            "baseline_threshold_v1",
        }:
            raise ValueError("Unknown routing.mode")
        self.threshold_multiplier = float(routing.get("threshold_multiplier", 1.5))
        self.baseline_threshold_numerator = float(
            routing.get("baseline_threshold_numerator", 2.0)
        )
        if self.baseline_threshold_numerator <= 0:
            raise ValueError("routing.baseline_threshold_numerator must be positive")
        self.selection_count = int(routing.get("selection_count", 1))
        if self.selection_count < 1 or self.threshold_multiplier < 0:
            raise ValueError("Invalid routing count/threshold")
        if self.threshold_margin_coef > 0 and self.routing_mode != "legacy_threshold":
            raise ValueError(
                "training.threshold_margin_coef requires routing.mode=legacy_threshold"
            )
        if self.threshold_margin_coef > 0 and self.threshold_multiplier <= 0:
            raise ValueError(
                "routing.threshold_multiplier must be positive when threshold-margin training is enabled"
            )
        if self.reward_mode == "baseline_compatible_v1" and self.threshold_margin_coef != 0:
            raise ValueError(
                "baseline_compatible_v1 requires training.threshold_margin_coef=0"
            )
        if (
            self.reward_mode == "baseline_compatible_v1"
            and self.routing_mode != "baseline_threshold_v1"
        ):
            raise ValueError(
                "baseline_compatible_v1 requires routing.mode=baseline_threshold_v1"
            )
        if (
            self.routing_mode == "baseline_threshold_v1"
            and self.reward_mode != "baseline_compatible_v1"
        ):
            raise ValueError(
                "baseline_threshold_v1 requires reward.mode=baseline_compatible_v1"
            )
        self.objective_version = "sum_discounted_leaf_v1"
        self.estimator_version = (
            "joint_episode_v2"
            if self.routing_mode == "categorical_set_v2"
            else (
                "baseline_threshold_surrogate_v1"
                if self.routing_mode == "baseline_threshold_v1"
                else "legacy_surrogate_v1"
            )
        )
        self.decisions = []
        self._path_lookup = {}
        self.audit = None
        self.cost_scale = self.step_penalty
        self.seed = int(self.config.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.state_representation = TaskAnalyzerRepresentation(
            self.runtime_config.get("task_analyzer", {})
        )
        reward_config = self.runtime_config.get("trajectory_reward")
        if reward_config is None:
            legacy = dict(self.runtime_config.get("reward_model", {}))
            reward_config = {
                **legacy,
                "enabled": bool(legacy.get("use_as_scalar_reward", False)),
                "backend": legacy.get("backend", "transformers_local"),
            }
            if legacy:
                logger.warning(
                    "reward_model is deprecated; use trajectory_reward instead"
                )
        reward_config = dict(reward_config or {})
        self.trajectory_reward_config = reward_config
        self.reward_weight = float(reward_config.get("weight", 0.1))
        self.reward_centered = bool(reward_config.get("centered", True))
        self.reward_failure_policy = str(
            reward_config.get("failure_policy", "task_reward_only")
        )
        if self.reward_failure_policy not in {"task_reward_only", "raise"}:
            raise ValueError(
                "trajectory_reward.failure_policy must be task_reward_only or raise"
            )
        apply_during = reward_config.get("apply_during", "train")
        apply_modes = (
            {str(apply_during)}
            if isinstance(apply_during, str)
            else {str(mode) for mode in apply_during}
        )
        self.trajectory_reward_active = bool(
            reward_config.get("enabled", False)
            and self.training
            and self.policy_mode == "train"
            and self.dataset_mode in apply_modes
        )
        if self.reward_mode == "baseline_compatible_v1" and reward_config.get("enabled", False):
            raise ValueError(
                "baseline_compatible_v1 requires trajectory_reward.enabled=false"
            )
        self.reward_scorer = (
            AuxiliaryRewardScorer(reward_config)
            if self.trajectory_reward_active
            else None
        )
        if (
            self.trajectory_reward_active
            and str(reward_config.get("backend", "")).lower()
            in {"http", "remote", "http_reward"}
            and not reward_config.get("base_url")
            and not os.getenv(
                reward_config.get(
                    "base_url_env", "SKYWORK_REWARD_BASE_URL"
                ),
                "",
            )
        ):
            logger.warning(
                "Skywork trajectory reward is enabled but its remote URL is "
                "not configured; terminal scoring will use task_reward_only"
            )
        self.candidate_dim = len(ROLE_NAMES) + 3 * len(CAPABILITY_DIMENSIONS) + 7
        self.policy_network = CandidateScoringPolicyNetwork(
            self.state_representation.dim, self.candidate_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.learning_rate
        )
        graph_config = self.runtime_config.get("graph", {})
        self.max_width = int(
            graph_config.get("max_width", graph_config.get("max_parallel_paths", 4))
        )
        self.max_depth = int(
            graph_config.get("max_depth", graph_config.get("max_step_num", 2))
        )
        if self.max_depth < 1:
            raise ValueError("graph.max_depth must be positive")
        self.agent_hash_list = agent_graph.hash_nodes
        self.agent_role_list = agent_graph.role_nodes
        self.baseline_terminator_index = agent_graph.terminator_agent_index
        if self.routing_mode == "baseline_threshold_v1":
            if self.baseline_terminator_index is None:
                raise ValueError(
                    "baseline_threshold_v1 requires a persona with the terminate action"
                )
            if not self.agent_graph.availability_mask[self.baseline_terminator_index]:
                raise ValueError(
                    "baseline_threshold_v1 requires its terminate persona to be available"
                )
        self.trajectories = []
        self.entropies = []
        self.rewards_history = []
        self.reward_breakdowns = []
        self.global_step = 0

        paths = self.config.get("paths", {})
        self.model_path = paths.get("model_path")
        if paths.get("load_policy") or self.loading:
            model_path = self.model_path or self.get_latest_model_path()
            if model_path:
                self.load_model(model_path)

    def _candidate_features(self):
        rows = []
        for view in self.agent_graph.public_agent_views():
            role = [1.0 if view["role_name"] == name else 0.0 for name in ROLE_NAMES]
            means = list(view["capability_mean"])
            uncertainty = list(view["uncertainty"])
            counts = [math.log1p(value) / 10.0 for value in view["observation_count"]]
            extras = [
                float(view["role_adherence"]),
                float(view["role_adherence_uncertainty"]),
                float(view["global_reliability"]),
                float(view["global_reliability_uncertainty"]),
                math.log1p(float(view["global_reliability_count"])) / 10.0,
                float(bool(view["tools"])),
                float(bool(view["available"])),
            ]
            rows.append(role + means + uncertainty + counts + extras)
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def get_state_representation(self, global_info):
        path_context = getattr(global_info, "path_context", None)
        if path_context is not None:
            # Match the baseline orchestrator input, reconstructed from this
            # path's sessions rather than shared teammate objects.
            context = path_context.baseline_orchestrator_messages(
                global_info.task.get("Question", "")
            )
        else:
            context = [{
                "role": "system",
                "content": "You are an assistant. Your task is to {}".format(
                    global_info.task.get("Question", "")
                ),
            }]
        state, _ = self.state_representation(context)
        return state.to(self.device)

    def _distribution(self, global_info, allow_stop):
        include_internal_stop = self.routing_mode != "baseline_threshold_v1"
        state = self.get_state_representation(global_info)
        probs = self.policy_network(
            state, self._candidate_features(), include_stop=include_internal_stop
        )
        mask_values = list(self.agent_graph.availability_mask)
        if include_internal_stop:
            mask_values.append(bool(allow_stop))
        mask = torch.tensor(mask_values, device=self.device, dtype=torch.bool)
        self._raw_probs = probs.detach().cpu().tolist()[0]
        self._action_mask = mask.detach().cpu().tolist()
        probs = probs.masked_fill(~mask.unsqueeze(0), 0.0)
        denominator = probs.sum(dim=-1, keepdim=True)
        if torch.any(denominator <= 0):
            raise RuntimeError("No available router action")
        probs = probs / denominator
        return probs

    def _choose(self, probs, allow_stop):
        distribution = torch.distributions.Categorical(probs)
        first_index = int(distribution.sample().item())
        self._sampled_first = first_index
        stop_index = len(self.agent_hash_list)
        if allow_stop and first_index == stop_index:
            self._threshold_candidates = []
            return [first_index], distribution
        available_count = max(1, sum(bool(v) for v in self.agent_graph.availability_mask))
        threshold = getattr(self, "threshold_multiplier", 1.5) / available_count
        candidates = torch.nonzero(probs[0, :-1] >= threshold).flatten().tolist()
        self._threshold_candidates = list(candidates)
        if first_index < stop_index and first_index not in candidates and not candidates:
            candidates.append(first_index)
        ordered = sorted(
            set(candidates),
            key=lambda i: probs[0, i].detach().item(),
            reverse=True,
        )
        return ordered[:self.max_width] or [first_index], distribution

    def _choose_baseline_threshold(self, probs, capacity):
        """Replicate the baseline threshold/fallback selector over persona agents.

        The baseline has no internal STOP action.  Its terminate persona remains an
        ordinary selectable agent, so the policy samples only the N agent logits.
        """
        agent_count = len(self.agent_hash_list)
        threshold = self.baseline_threshold_numerator / agent_count
        agent_probs = probs[0, :agent_count]
        candidates = torch.nonzero(agent_probs > threshold).flatten().tolist()
        self._threshold_candidates = list(candidates)
        distribution = torch.distributions.Categorical(agent_probs)
        if candidates:
            indices = sorted(
                candidates,
                key=lambda index: agent_probs[index].detach().item(),
                reverse=True,
            )
        else:
            available_count = int(torch.count_nonzero(agent_probs > 0).item())
            sample_count = min(capacity, self.max_width, available_count)
            if sample_count < 1:
                raise RuntimeError("No available agent for baseline threshold routing")
            indices = torch.multinomial(
                agent_probs, sample_count, replacement=False
            ).tolist()
        self._sampled_first = indices[0] if indices else None
        return indices[: min(capacity, self.max_width)], distribution, threshold

    def begin_task(self, audit=None):
        self.trajectories = []
        self.entropies = []
        self.decisions = []
        self._path_lookup = {}
        self.audit = audit

    def abort_task(self):
        self.begin_task(getattr(self, "audit", None))

    def propose(self, global_info, capacity):
        if capacity < 1:
            raise ValueError("Routing requires reserved capacity")
        baseline_sampling = self.routing_mode == "baseline_threshold_v1"
        allow_stop = global_info.path_id != -1 and not baseline_sampling
        from contextlib import nullcontext
        scope = (self.audit.call_scope(path_uid=getattr(global_info, "path_uid", "root"),
                                      purpose="task_encoder") if self.audit else nullcontext())
        with scope:
            probs = self._distribution(global_info, allow_stop)
        stop_index = len(self.agent_hash_list) if not baseline_sampling else None
        if self.routing_mode == "categorical_set_v2":
            selection = sample_ordered(probs, min(capacity, self.selection_count), stop_index)
            indices = selection.indices
            joint = selection.log_prob
            entropy = selection.entropy
            conditional = selection.conditional_probs
            threshold = None
        elif baseline_sampling:
            indices, distribution, threshold = self._choose_baseline_threshold(
                probs, capacity
            )
            joint = None
            entropy = distribution.entropy().mean()
            conditional = None
        else:
            indices, distribution = self._choose(probs, allow_stop)
            joint = None
            entropy = distribution.entropy().mean()
            conditional = None
            count = max(1, sum(bool(v) for v in self.agent_graph.availability_mask))
            threshold = self.threshold_multiplier / count
        decision_id = self.audit.new_id("decision") if self.audit else f"decision-{len(self.decisions)}"
        actions = [
            ORCHESTRATOR_STOP if stop_index is not None and i == stop_index
            else self.agent_hash_list[i]
            for i in indices
        ]
        proposal = dict(decision_id=decision_id, actions=actions, indices=indices, probs=probs,
                        joint_log_prob=joint, entropy=entropy, capacity=capacity)
        if self.audit:
            self.audit.emit("routing_decision", decision_id=decision_id,
                path_uid=getattr(global_info, "path_uid", "root"), allow_stop=allow_stop,
                p_stop=(float(probs[0, stop_index].detach()) if stop_index is not None else None),
                probabilities=probs[0], raw_probabilities=self._raw_probs, mask=self._action_mask,
                capacity=capacity, steps_completed=len(global_info.workflow.workflow),
                remaining_depth=getattr(global_info, "remaining_depth", None),
                mode=self.routing_mode, entropy=entropy, selected=actions,
                threshold=threshold,
                sampled_first=getattr(self, "_sampled_first", None) if joint is None else indices[0],
                threshold_candidates=getattr(self, "_threshold_candidates", []) if joint is None else [],
                conditional_probs=conditional, joint_log_prob=joint,
                estimator=self.estimator_version)
        return proposal

    @staticmethod
    def _threshold_surplus_for_indices(probs, indices, threshold, cap):
        """Return a differentiable, capped log-ratio above the legacy threshold."""
        if not indices:
            return probs.new_zeros(())
        selected_probs = probs[0, indices]
        return torch.log(selected_probs / threshold).clamp(min=0.0, max=cap).mean()

    def _selected_threshold_surplus(self, proposal, selected_indices):
        if self.threshold_margin_coef <= 0:
            return None
        stop_index = len(self.agent_hash_list)
        non_stop_indices = [index for index in selected_indices if index != stop_index]
        available_count = max(1, sum(bool(v) for v in self.agent_graph.availability_mask))
        threshold = self.threshold_multiplier / available_count
        return self._threshold_surplus_for_indices(
            proposal["probs"], non_stop_indices, threshold, self.threshold_margin_cap
        )

    def accept(self, proposal, parent_uid, allocations):
        accepted = [item["action"] for item in allocations]
        if self.routing_mode == "categorical_set_v2" and accepted != proposal["actions"]:
            raise RuntimeError("Sampling/execution mismatch: capacity must be reserved before sampling")
        if not allocations:
            return
        selected_indices = [
            len(self.agent_hash_list) if item["action"] == ORCHESTRATOR_STOP
            else self.agent_hash_list.index(item["action"])
            for item in allocations
        ]
        threshold_surplus = self._selected_threshold_surplus(proposal, selected_indices)
        parent_index = self._path_lookup.get(parent_uid)
        prefix = ([dict(step) for step in self.trajectories[parent_index]]
                  if parent_index is not None else [])
        for item in allocations:
            uid, action = item["path_uid"], item["action"]
            if uid not in self._path_lookup:
                self._path_lookup[uid] = len(self.trajectories)
                self.trajectories.append([dict(step) for step in prefix])
        for item in allocations:
            action = item["action"]
            index = len(self.agent_hash_list) if action == ORCHESTRATOR_STOP else self.agent_hash_list.index(action)
            log_prob = torch.log(proposal["probs"][0, index]).reshape(1)
            initial_reward = (
                0.0
                if self.reward_mode == "baseline_compatible_v1"
                else (0.0 if action == ORCHESTRATOR_STOP else -self.step_penalty)
            )
            self.trajectories[self._path_lookup[item["path_uid"]]].append(dict(
                log_prob=log_prob, action=action, action_id=item["action_id"],
                decision_id=proposal["decision_id"], reward=initial_reward,
                finalized=False))
        self.entropies.append(proposal["entropy"])
        self.decisions.append(dict(
            decision_id=proposal["decision_id"],
            log_prob=proposal["joint_log_prob"],
            actions=[dict(item) for item in allocations],
            threshold_surplus=threshold_surplus,
        ))

    def forward(self, global_info):
        # Compatibility API; the runtime uses propose/accept with stable path IDs.
        proposal = self.propose(global_info, self.max_width)
        parent = getattr(global_info, "path_uid", str(global_info.path_id))
        allocations = [dict(path_uid=(str(i) if global_info.path_id == -1 else
                        parent if i == 0 else f"{parent}/{len(self.trajectories)+i}"),
                        action=a, action_id=f"{proposal['decision_id']}/{i}")
                       for i, a in enumerate(proposal["actions"])]
        self.accept(proposal, parent, allocations)
        return proposal["actions"]

    @staticmethod
    def _json_safe(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            return value.item() if value.numel() == 1 else value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): RoleAwareREINFORCE._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [RoleAwareREINFORCE._json_safe(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _trajectory_quality(self, transition, global_info):
        result = {
            "skywork_raw": None,
            "skywork_centered": None,
            "skywork_weighted": 0.0,
            "skywork_status": "disabled",
            "trajectory_hash": None,
            "skywork_error": None,
        }
        if not self.trajectory_reward_active or self.reward_scorer is None:
            return result

        reward_input = build_trajectory_reward_input(
            task=global_info.task,
            workflow=global_info.workflow,
            candidate_output=transition.get("candidate_output", ""),
            artifact_root=global_info.workpath,
            max_chars=int(self.trajectory_reward_config.get("max_chars", 12000)),
            max_step_chars=int(
                self.trajectory_reward_config.get("max_step_chars", 3000)
            ),
        )
        result["trajectory_hash"] = reward_input.digest
        try:
            raw = float(self.reward_scorer(reward_input.messages))
            quality_value = 2.0 * raw - 1.0 if self.reward_centered else raw
            result.update(
                {
                    "skywork_raw": raw,
                    "skywork_centered": quality_value,
                    "skywork_weighted": self.reward_weight * quality_value,
                    "skywork_status": "ok",
                }
            )
        except Exception as error:
            if self.reward_failure_policy == "raise":
                raise
            result["skywork_status"] = "failed"
            result["skywork_error"] = str(error)[:500]
            logger.warning(
                "Skywork trajectory scoring failed; using task reward only: %s",
                error,
            )
        return result

    def _write_reward_trace(self, global_info, breakdown):
        try:
            os.makedirs(global_info.workpath, exist_ok=True)
            target = os.path.join(
                global_info.workpath,
                self.trajectory_reward_config.get(
                    "trace_file", "trajectory_rewards.jsonl"
                ),
            )
            with open(target, "a", encoding="utf-8") as destination:
                json.dump(
                    self._json_safe(breakdown),
                    destination,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                destination.write("\n")
        except OSError as error:
            logger.warning("Could not write trajectory reward trace: %s", error)

    def _baseline_step_scale(self, step_index):
        """Return the baseline logarithmic step scale for a zero-based action step."""
        config = self.baseline_reward_config
        scale = float(config.get("step_scale", 0.1))
        growth_rate = float(config.get("growth_rate", 1.0))
        if growth_rate <= 0:
            raise ValueError("baseline_compatible.growth_rate must be positive")
        normalized_step = (int(step_index) + 1) / (self.max_depth + 1)
        value = scale * math.log(1.0 + growth_rate * normalized_step) / math.log(
            1.0 + growth_rate
        )
        return scale - value if bool(config.get("inverse", False)) else value

    def _baseline_action_factor(self, runtime_action):
        payload = getattr(runtime_action, "action", {}) or {}
        action_name = str(payload.get("action", "")).strip().lower()
        web_actions = {
            str(name).strip().lower()
            for name in self.baseline_reward_config.get(
                "web_actions",
                ["search_bing", "search_arxiv", "access_website"],
            )
        }
        if action_name in web_actions:
            return float(self.baseline_reward_config.get("web_action_factor", -1.5))
        return float(self.baseline_reward_config.get("default_action_factor", -1.0))

    @staticmethod
    def _runtime_action_name(runtime_action):
        payload = getattr(runtime_action, "action", {}) or {}
        return str(payload.get("action", "")).strip().lower()

    def _finalize_baseline_compatible(self, transition, global_info, trajectory, path_id):
        """Assign baseline reward terms without changing the role-aware router."""
        actions_by_id = {
            getattr(action, "action_id", None): action
            for action in global_info.workflow.workflow
        }
        normalization = float(
            self.baseline_reward_config.get("action_cost_normalization", 100000.0)
        )
        if normalization <= 0:
            raise ValueError(
                "baseline_compatible.action_cost_normalization must be positive"
            )

        action_terms = []
        executed_steps = 0
        for index, step in enumerate(trajectory):
            if step["action"] == ORCHESTRATOR_STOP:
                continue
            runtime_action = actions_by_id.get(step.get("action_id"))
            if runtime_action is None:
                # A missing runtime action can only occur for an internal STOP.  Keep
                # the transition neutral rather than assigning another path's cost.
                step["reward"] = 0.0
                continue
            action_name = self._runtime_action_name(runtime_action)
            if action_name == "terminate":
                step["reward"] = 0.0
                continue
            scale = self._baseline_step_scale(executed_steps)
            factor = self._baseline_action_factor(runtime_action)
            action_cost = float(getattr(runtime_action, "cost", 0.0))
            reward = factor * scale * action_cost / normalization
            step["reward"] = reward
            action_terms.append(
                {
                    "trajectory_index": index,
                    "action_id": step.get("action_id"),
                    "action_name": action_name,
                    "step_scale": scale,
                    "action_cost": action_cost,
                    "factor": factor,
                    "reward": reward,
                }
            )
            executed_steps += 1

        task_reward = float(transition.get("reward", 0.0))
        terminal_factor = float(
            self.baseline_reward_config.get("terminal_factor", 0.5)
        )
        # ContinuousREINFORCE calls logarithmic_cost(len(trajectory)): this
        # includes an explicit terminator when one was selected.
        terminal_scale = self._baseline_step_scale(len(trajectory))
        terminal_reward = (
            task_reward + terminal_factor * terminal_scale
            if task_reward > 0
            else task_reward - terminal_factor * terminal_scale
        )

        # A selected internal STOP has no runtime Action.  At a depth limit the
        # terminal term is attached to the last executed action, preserving the
        # role-aware runtime while exposing the same scalar reward components.
        trajectory[-1]["reward"] += terminal_reward
        trajectory[-1]["finalized"] = True
        trajectory[-1]["total_tokens"] = global_info.total_tokens
        trajectory[-1]["total_cost"] = global_info.total_cost
        trajectory[-1]["metrics"] = transition.get("metrics", {})
        trajectory[-1]["candidate_output"] = transition.get("candidate_output", "")
        path_reward = sum(float(step["reward"]) for step in trajectory)
        action_penalty_total = -sum(term["reward"] for term in action_terms)

        breakdown = {
            "reward_mode": self.reward_mode,
            "path_id": path_id,
            "task_reward": task_reward,
            "skywork_raw": None,
            "skywork_centered": None,
            "skywork_weighted": 0.0,
            "skywork_status": "disabled",
            "action_reward_terms": action_terms,
            "executed_steps": executed_steps,
            "terminal_scale": terminal_scale,
            "terminal_reward": terminal_reward,
            "step_penalty": action_penalty_total,
            "normalized_token_cost": 0.0,
            "token_cost_penalty": 0.0,
            "path_reward": path_reward,
            "total_tokens": global_info.total_tokens,
            "total_cost": global_info.total_cost,
            "metrics": transition.get("metrics", {}),
            "scorer_model": None,
        }
        trajectory[-1]["reward_breakdown"] = breakdown
        self.rewards_history.append(path_reward)
        self.reward_breakdowns.append(breakdown)
        self._write_reward_trace(global_info, breakdown)
        if getattr(self, "audit", None):
            self.audit.emit(
                "reward_assigned",
                path_uid=transition.get("path_uid"),
                decision_ids=[step.get("decision_id") for step in trajectory],
                action_ids=[step.get("action_id") for step in trajectory],
                **breakdown,
            )

    def finalize_task(self, transition, global_info):
        path_id = getattr(self, "_path_lookup", {}).get(
            transition.get("path_uid"), int(transition.get("path_id", 0)))
        if path_id >= len(self.trajectories) or not self.trajectories[path_id]:
            return
        trajectory = self.trajectories[path_id]
        if self.reward_mode == "baseline_compatible_v1":
            self._finalize_baseline_compatible(
                transition, global_info, trajectory, path_id
            )
            return
        task_reward = float(transition.get("reward", 0.0))
        normalized_cost = float(global_info.total_cost) / self.cost_normalization
        token_cost_penalty = self.token_cost_weight * normalized_cost
        step_penalty_total = -sum(float(step["reward"]) for step in trajectory)
        from contextlib import nullcontext
        scope = (self.audit.call_scope(path_uid=transition.get("path_uid"), purpose="reward_scorer")
                 if getattr(self, "audit", None) else nullcontext())
        with scope:
            quality = self._trajectory_quality(transition, global_info)
        terminal_reward = (
            task_reward
            + float(quality["skywork_weighted"])
            - token_cost_penalty
        )
        trajectory[-1]["reward"] += terminal_reward
        trajectory[-1]["finalized"] = True
        trajectory[-1]["total_tokens"] = global_info.total_tokens
        trajectory[-1]["total_cost"] = global_info.total_cost
        trajectory[-1]["metrics"] = transition.get("metrics", {})
        trajectory[-1]["candidate_output"] = transition.get("candidate_output", "")
        path_reward = sum(float(step["reward"]) for step in trajectory)

        breakdown = {
            "path_id": path_id,
            "task_reward": task_reward,
            **quality,
            "step_penalty": step_penalty_total,
            "normalized_token_cost": normalized_cost,
            "token_cost_penalty": token_cost_penalty,
            "terminal_reward": terminal_reward,
            "path_reward": path_reward,
            "total_tokens": global_info.total_tokens,
            "total_cost": global_info.total_cost,
            "metrics": transition.get("metrics", {}),
            "scorer_model": (
                self.trajectory_reward_config.get("model")
                if self.trajectory_reward_active
                else None
            ),
        }
        trajectory[-1]["reward_breakdown"] = breakdown
        self.rewards_history.append(path_reward)
        self.reward_breakdowns.append(breakdown)
        self._write_reward_trace(global_info, breakdown)
        if getattr(self, "audit", None):
            self.audit.emit("reward_assigned", path_uid=transition.get("path_uid"),
                            decision_ids=[step.get("decision_id") for step in trajectory],
                            action_ids=[step.get("action_id") for step in trajectory], **breakdown)

    def _returns(self, trajectory):
        result = []
        value = 0.0
        for step in reversed(trajectory):
            value = float(step["reward"]) + self.gamma * value
            result.insert(0, value)
        return torch.tensor(result, dtype=torch.float32, device=self.device)

    def update(self):
        completed = [
            trajectory for trajectory in self.trajectories
            if trajectory and trajectory[-1].get("finalized")
        ]
        if not completed:
            return {}
        losses = []
        returns_all = []
        for trajectory in completed:
            returns = self._returns(trajectory)
            returns_all.extend(returns.detach().cpu().tolist())
            for step, value in zip(trajectory, returns):
                losses.append(-step["log_prob"] * value)
        if getattr(self, "routing_mode", "legacy_threshold") == "categorical_set_v2":
            if len(completed) != len(self.trajectories):
                raise RuntimeError("Cannot update an incomplete episode")
            task_return = sum(self._returns(t)[0] for t in completed).detach()
            loss = -torch.stack([d["log_prob"] for d in self.decisions]).sum() * task_return
        else:
            task_return = sum(self._returns(t)[0] for t in completed).detach()
            loss = torch.stack(losses).sum()
        if self.entropies:
            loss -= self.entropy_coef * torch.stack(self.entropies).sum()
        threshold_surpluses = [
            decision["threshold_surplus"]
            for decision in self.decisions
            if decision.get("threshold_surplus") is not None
        ]
        if threshold_surpluses:
            threshold_margin_mean = torch.stack(threshold_surpluses).mean()
            threshold_margin_loss = -self.threshold_margin_coef * threshold_margin_mean
            loss += threshold_margin_loss
        else:
            threshold_margin_mean = loss.new_zeros(())
            threshold_margin_loss = loss.new_zeros(())
        optimizer_stepped = False
        gradient_norm = None
        if self.optimizer_updates_enabled:
            self.optimizer.zero_grad()
            loss.backward()
            gradient_norm = float(torch.sqrt(sum(
                (p.grad.detach() ** 2).sum() for p in self.policy_network.parameters() if p.grad is not None)))
            self.optimizer.step()
            self.global_step += 1
            optimizer_stepped = True
        metrics = {
            "policy_loss": float(loss.detach().cpu().item()),
            "mean_reward": float(np.mean(returns_all)),
            "optimizer_stepped": optimizer_stepped,
            "task_return": float(task_return), "return_variance": float(np.var(returns_all)),
            "gradient_norm": gradient_norm,
            "threshold_margin_mean": float(threshold_margin_mean.detach().cpu().item()),
            "threshold_margin_loss": float(threshold_margin_loss.detach().cpu().item()),
            "estimator": getattr(self, "estimator_version", "legacy_surrogate_v1"),
        }
        if getattr(self, "audit", None):
            self.audit.emit("policy_update", decision_ids=[d["decision_id"] for d in self.decisions], **metrics)
        self.trajectories = []
        self.entropies = []
        self.decisions = []
        self._path_lookup = {}
        return metrics

    def training_state_dict(self):
        return {
            "runtime_version": "path_sessions_v1",
            "routing_mode": self.routing_mode,
            "routing_version": "2",
            "objective_version": self.objective_version,
            "estimator_version": self.estimator_version,
            "reward_mode": self.reward_mode,
            "model_state_dict": self.policy_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.policy_network.state_dim,
            "candidate_dim": self.policy_network.candidate_dim,
            "roles": ROLE_NAMES,
            "global_step": self.global_step,
            "config": self.config,
        }

    def load_training_state_dict(self, checkpoint, load_optimizer=True):
        if checkpoint.get("state_dim") != self.policy_network.state_dim:
            raise ValueError("Checkpoint state dimension mismatch")
        if checkpoint.get("candidate_dim") != self.policy_network.candidate_dim:
            raise ValueError("Checkpoint candidate feature dimension mismatch")
        checkpoint_roles = checkpoint.get("roles")
        if checkpoint_roles is not None and tuple(checkpoint_roles) != tuple(
            ROLE_NAMES
        ):
            raise ValueError("Checkpoint role schema mismatch")
        if load_optimizer and checkpoint.get("runtime_version") != "path_sessions_v1":
            raise ValueError("Runtime version mismatch; use explicit weights-only evaluation")
        source_mode = checkpoint.get("routing_mode", "legacy_threshold")
        target_mode = getattr(self, "routing_mode", "legacy_threshold")
        transfer = getattr(self, "config", {}).get("routing", {}).get("allow_policy_transfer", False)
        if source_mode != target_mode and (load_optimizer or not transfer):
            raise ValueError("Routing version mismatch; weights-only transfer requires explicit allow_policy_transfer")
        if load_optimizer and checkpoint.get("estimator_version", "legacy_surrogate_v1") != getattr(self, "estimator_version", "legacy_surrogate_v1"):
            raise ValueError("Estimator version mismatch")
        if load_optimizer and checkpoint.get("objective_version") != self.objective_version:
            raise ValueError("Objective version mismatch")
        source_reward_mode = checkpoint.get("reward_mode", "role_aware_v1")
        if load_optimizer and source_reward_mode != self.reward_mode:
            raise ValueError(
                "Reward mode mismatch; start a new run or use explicit weights-only evaluation"
            )
        if load_optimizer and checkpoint.get("routing_version") != "2":
            raise ValueError("Routing version mismatch")
        self.policy_network.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if load_optimizer:
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is None:
                raise ValueError("Train resume checkpoint is missing optimizer state")
            self.optimizer.load_state_dict(optimizer_state)
        self.global_step = int(checkpoint.get("global_step", 0))

    def save_model(self, path=None, tag=None):
        checkpoint_dir = path or self.config.get("paths", {}).get(
            "checkpoint_path", "checkpoint/role_aware"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""
        target = os.path.join(checkpoint_dir, f"role_router_{timestamp}{suffix}.pt")
        torch.save(self.training_state_dict(), target)
        return target

    def load_model(self, path, strict=True):
        checkpoint = torch.load(path, map_location=self.device)
        if "policy" in checkpoint:
            checkpoint = checkpoint["policy"]
        self.load_training_state_dict(checkpoint, load_optimizer=self.training)
        return True

    def get_latest_model_path(self):
        checkpoint_dir = self.config.get("paths", {}).get("checkpoint_path", "")
        if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
            return None
        files = [
            os.path.join(checkpoint_dir, name)
            for name in os.listdir(checkpoint_dir)
            if name.endswith(".pt")
        ]
        return max(files, key=os.path.getctime) if files else None
