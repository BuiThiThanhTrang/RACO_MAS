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
        self.agent_hash_list = agent_graph.hash_nodes
        self.agent_role_list = agent_graph.role_nodes
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
        context = [
            {
                "role": "user",
                "content": str(global_info.task.get("Question", "")),
            }
        ]
        if global_info.workflow.workflow:
            context.append(
                {
                    "role": "assistant",
                    "content": global_info.workflow.language_state,
                }
            )
        state, _ = self.state_representation(context)
        return state.to(self.device)

    def _distribution(self, global_info, allow_stop):
        state = self.get_state_representation(global_info)
        probs = self.policy_network(state, self._candidate_features(), include_stop=True)
        mask = torch.tensor(
            list(self.agent_graph.availability_mask) + [bool(allow_stop)],
            device=self.device,
            dtype=torch.bool,
        )
        probs = probs.masked_fill(~mask.unsqueeze(0), 0.0)
        denominator = probs.sum(dim=-1, keepdim=True)
        if torch.any(denominator <= 0):
            raise RuntimeError("No available router action")
        probs = probs / denominator
        return probs

    def _choose(self, probs, allow_stop):
        distribution = torch.distributions.Categorical(probs)
        if self.training:
            first = distribution.sample()
        else:
            first = distribution.sample()
        first_index = int(first.item())
        stop_index = len(self.agent_hash_list)
        if allow_stop and first_index == stop_index:
            return [first_index], distribution

        available_count = max(
            1,
            sum(bool(available) for available in self.agent_graph.availability_mask),
        )
        threshold = 1.0 / available_count
        candidates = torch.nonzero(probs[0, :-1] >= threshold).flatten()
        if first_index < stop_index and first_index not in candidates.tolist():
            candidates = torch.cat((torch.tensor([first_index], device=self.device), candidates))
        ordered = sorted(
            set(int(index) for index in candidates.tolist()),
            key=lambda index: float(probs[0, index]),
            reverse=True,
        )
        return ordered[: self.max_width] or [first_index], distribution

    def forward(self, global_info):
        allow_stop = global_info.path_id != -1
        probs = self._distribution(global_info, allow_stop)
        selected, distribution = self._choose(probs, allow_stop)
        entropy = distribution.entropy().mean()
        self.entropies.append(entropy)
        stop_index = len(self.agent_hash_list)

        if global_info.path_id == -1:
            while len(self.trajectories) < len(selected):
                self.trajectories.append([])
            path_indices = list(range(len(selected)))
        else:
            while len(self.trajectories) <= global_info.path_id:
                self.trajectories.append([])
            path_indices = [global_info.path_id]
            for _ in selected[1:]:
                self.trajectories.append(list(self.trajectories[global_info.path_id]))
                path_indices.append(len(self.trajectories) - 1)

        outputs = []
        for path_index, action_index in zip(path_indices, selected):
            action_tensor = torch.tensor(action_index, device=self.device)
            action_name = (
                ORCHESTRATOR_STOP
                if action_index == stop_index
                else self.agent_hash_list[action_index]
            )
            self.trajectories[path_index].append(
                {
                    "log_prob": distribution.log_prob(action_tensor).reshape(1),
                    "action": action_name,
                    "reward": (
                        0.0
                        if action_name == ORCHESTRATOR_STOP
                        else -self.step_penalty
                    ),
                    "finalized": False,
                }
            )
            outputs.append(action_name)
        return outputs

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

    def finalize_task(self, transition, global_info):
        path_id = int(transition.get("path_id", 0))
        if path_id >= len(self.trajectories) or not self.trajectories[path_id]:
            return
        trajectory = self.trajectories[path_id]
        task_reward = float(transition.get("reward", 0.0))
        normalized_cost = float(global_info.total_cost) / self.cost_normalization
        token_cost_penalty = self.token_cost_weight * normalized_cost
        step_penalty_total = -sum(float(step["reward"]) for step in trajectory)
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
        loss = torch.stack(losses).sum()
        if self.entropies:
            loss -= self.entropy_coef * torch.stack(self.entropies).sum()
        optimizer_stepped = False
        if self.optimizer_updates_enabled:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            optimizer_stepped = True
        metrics = {
            "policy_loss": float(loss.detach().cpu().item()),
            "mean_reward": float(np.mean(returns_all)),
            "optimizer_stepped": optimizer_stepped,
        }
        self.trajectories = []
        self.entropies = []
        return metrics

    def training_state_dict(self):
        return {
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
