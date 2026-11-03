from typing import List
from pathlib import Path
from contextlib import nullcontext
from role_aware.audit_trace import AuditTrace, digest, redact
from role_aware.aggregation import (
    aggregate_candidates,
    aggregate_gaia_candidates,
    aggregate_musique_candidates,
    normalize_choice,
    select_gaia_path_answer,
    select_srdd_artifact,
)
import json
import os
import copy
import logging

from inference.reasoning.path import ReasoningState, GraphReasoningPath
from inference.graph.agent_graph import AgentGraph
from inference.graph.action_graph import ActionGraph

from utils.logging import LogManager

from agent.agent_info.global_info import GlobalInfo

from tasks.evaluator import BenchmarkEvaluator
from role_aware.evidence import PathTerminalEvidence
from role_aware.route_experience import RouteExperienceEvent
from role_aware.musique_llm_canonicalizer import original_question
from role_aware.multiagentbench_cs import (
    build_router_cs_semantic_trace,
    judge_router_cs_semantic,
)

main_logger = logging.getLogger('global')

class GraphReasoning:
    def __init__(self, task:json, graph: AgentGraph, policy, action_graph, max_parallel_paths, max_step_num, runtime_config, registry, profile_evidence=None, external_tools_enabled=False, env=None, env_name=None, audit=None):
        self.task = task
        self.runtime_config = copy.deepcopy(dict(runtime_config))
        self.agent_graph = graph
        self.action_graph = action_graph
        self.reasoning_paths: List[GraphReasoningPath] = []

        self.max_parallel_paths = max_parallel_paths
        self.max_step_num = max_step_num
        self.registry = registry
        self.profile_evidence = profile_evidence

        self.final_answer = ""
        self.answers = []
        self._musique_semantic_cache = {}

        self.global_logger = LogManager("./config/global.yaml", self.task.get("type"),
                                        folder_path=audit.directory if audit else None)
        self.audit = audit or AuditTrace(self.global_logger.folder_path, "standalone",
                                        task.get("id", "unknown"), "attempt-1", enabled=False)

        self.workspace_path = self.global_logger.folder_path
        self.policy = policy
        self.policy.action_graph = self.action_graph
        self.external_tools_enabled = external_tools_enabled

        self.env = env
        self.env_name = env_name
        main_logger.info("{}[Graph Reasoning Initialized]{}".format("-"*30, "-"*30))
        main_logger.info(redact(self.runtime_config))
        main_logger.info(self.agent_graph.role_nodes)

    def save_checkpoint(self, save_data):
        main_logger.info("{}[Save Checkpoint]{}".format("-"*30, "-"*30))
        cur_acc = save_data["best_acc"]
        cur_data_len =  save_data["best_data_len"]
        main_logger.info("best acc: {}, data len: {}".format(cur_acc, cur_data_len))
        tag = "acc_{}-data_{}".format(cur_acc, cur_data_len)
        self.policy.save_model(path=None, tag=tag)

    def start(self, save_data):
        if save_data is not None:
            self.save_checkpoint(save_data)
        if hasattr(self.policy, "begin_task"):
            self.policy.begin_task(self.audit)
        self.audit.emit("task_started", split=self.runtime_config.get("audit_split", "unknown"),
                        max_width=self.max_parallel_paths, max_depth=self.max_step_num)
        try:
            self._route(None)
        except BaseException as error:
            if hasattr(self.policy, "abort_task"):
                self.policy.abort_task()
            self.audit.emit("task_interrupted", error_type=type(error).__name__)
            raise

    def _new_info(self, index):
        evaluation_only = {
            "Answer",
            "answer",
            "target",
            "answer_aliases",
            "gold_decomposition",
            "supporting_paragraph_indices",
        }
        public_task = {
            key: value for key, value in self.task.items() if key not in evaluation_only
        }
        info = GlobalInfo(path_id=index, workpath=self.workspace_path, task=public_task,
                          env=self.env, env_name=self.env_name)
        info.audit_split = self.runtime_config.get("audit_split", "unknown")
        return info

    def _binary_task_reward(self, correct):
        """Return the configured terminal reward for binary-answer tasks.

        Defaults preserve existing experiments: correct=+1 and incorrect=-1.
        The setting lives in runtime config so an experiment can use an
        incorrect reward of 0 without changing historical runs.
        """
        reward_config = self.runtime_config.get("task_reward", {})
        key = "correct" if correct else "incorrect"
        default = 1.0 if correct else -1.0
        return float(reward_config.get(key, default))

    def _musique_semantic_config(self):
        musique = self.runtime_config.get("musique", {}) or {}
        return dict(musique.get("semantic_outcome") or {})

    def _musique_semantic_enabled_for(self, purpose):
        config = self._musique_semantic_config()
        if not bool(config.get("enabled", False)):
            return False
        use_for = {
            str(value)
            for value in config.get(
                "use_for",
                ("reporting", "route_experience", "profile_evidence", "training_reward"),
            )
        }
        return str(purpose) in use_for

    def _judge_musique_post_task(
        self,
        snapshot,
        prediction,
        golds,
        *,
        official_success,
    ):
        """Run one post-commit call for router CS and semantic correctness."""

        config = self._musique_semantic_config()
        if not bool(config.get("enabled", False)):
            return {
                "verdict": "NOT_EVALUATED",
                "equivalent": False,
                "effective_equivalent": official_success,
                "confidence": None,
                "reason": "Semantic outcome evaluation is disabled.",
                "model": None,
                "tokens": 0,
                "llm_called": False,
                "collaboration_evaluated": False,
            }
        model = str(config.get("model") or "")
        if not model:
            raise ValueError("musique.semantic_outcome.model is required when enabled")
        cache_key = digest(
            {
                "prediction": str(prediction or ""),
                "golds": [str(value) for value in golds],
                "question": original_question(self.task.get("Question", "")),
                "model": model,
                "candidate_hash": snapshot.get("candidate_hash"),
                "prompt_version": "router_cs_semantic_v2",
            }
        )
        if cache_key in self._musique_semantic_cache:
            return dict(self._musique_semantic_cache[cache_key])
        trace = build_router_cs_semantic_trace(
            snapshot,
            self.audit.events,
            prediction=prediction,
            accepted_answers=golds,
            question=original_question(self.task.get("Question", "")),
        )
        try:
            with self.audit.call_scope(
                path_uid="task",
                purpose="musique_router_cs_semantic_judge",
            ):
                result, tokens = judge_router_cs_semantic(
                    trace,
                    model=model,
                    reasoning_effort=str(config.get("reasoning_effort", "low")),
                    max_repair_attempts=int(config.get("max_repair_attempts", 1)),
                    official_success=bool(official_success),
                )
            effective = bool(result.get("semantic_correct"))
            resolved = dict(
                result,
                verdict=result.get("answer_verdict"),
                equivalent=bool(result.get("answer_equivalent")),
                effective_equivalent=effective,
                confidence=result.get("answer_confidence"),
                reason=result.get("answer_rationale"),
                llm_called=True,
                collaboration_evaluated=True,
            )
            self.audit.emit(
                "musique_router_cs_semantic_judged",
                prediction_digest=digest(str(prediction or "")),
                verdict=resolved.get("verdict"),
                equivalent=resolved.get("equivalent"),
                effective_equivalent=effective,
                confidence=resolved.get("confidence"),
                routing_planning_score=resolved.get("routing_planning_score"),
                state_handoff_communication_score=resolved.get(
                    "state_handoff_communication_score"
                ),
                collaboration_score_100=resolved.get("collaboration_score_100"),
                model=model,
                tokens=int(tokens),
            )
        except Exception as error:
            if str(config.get("failure_policy", "exact_match")) == "raise":
                raise
            resolved = {
                "verdict": "ERROR_FALLBACK",
                "equivalent": False,
                "effective_equivalent": official_success,
                "confidence": None,
                "reason": f"Semantic judge failed closed: {type(error).__name__}.",
                "model": model,
                "tokens": 0,
                "llm_called": True,
                "collaboration_evaluated": False,
                "error_type": type(error).__name__,
            }
            self.audit.emit(
                "musique_router_cs_semantic_failed",
                prediction_digest=digest(str(prediction or "")),
                error_type=type(error).__name__,
                model=model,
            )
        self._musique_semantic_cache[cache_key] = dict(resolved)
        return dict(resolved)

    def _route(self, path):
        capacity = self.max_parallel_paths - len(self.reasoning_paths) + (1 if path else 0)
        if capacity < 1:
            raise RuntimeError("No capacity reserved for current path")
        info = path.global_info if path else self._new_info(-1)
        info.remaining_depth = self.max_step_num - (path.completed_steps if path else 0)
        info.path_uid = path.path_uid if path else "root"
        if hasattr(self.policy, "propose"):
            proposal = self.policy.propose(info, capacity)
        else:
            proposal = dict(decision_id=self.audit.new_id("decision"), actions=self.policy.forward(info))
            self.audit.emit("routing_decision", decision_id=proposal["decision_id"],
                            path_uid=info.path_uid, mode="fixed", p_stop=None,
                            selected=proposal["actions"], capacity=capacity)
        decision_id = proposal["decision_id"]
        requested = proposal["actions"]
        requested_assignments = proposal.get("assignments") or [None] * len(requested)
        stop = "__orchestrator_stop__"
        if stop in requested and (not path or requested != [stop]):
            raise RuntimeError("STOP must be a sole action on an existing path")
        accepted = requested[:capacity]
        invalid = [a for a in accepted if a != stop and self.registry.get_agent_from_idx(a) is None]
        if invalid or not accepted:
            self.audit.emit("allocation", decision_id=decision_id, accepted=[], rejected=requested,
                            reason="no_valid_agent", path_uid=info.path_uid)
            if path:
                path.finish("no_valid_agent")
            raise RuntimeError("Router returned unavailable/empty action")
        allocations = []
        new_paths = []
        for slot, action in enumerate(accepted):
            action_id = self.audit.new_id("action")
            if slot == 0 and path:
                target = path
            else:
                uid = self.audit.new_id("path")
                index = len(self.reasoning_paths) + len(new_paths)
                agent = self.registry.get_agent_from_idx(action)
                if path:
                    target = path.fork(index, uid, agent, self.workspace_path)
                else:
                    target = GraphReasoningPath(agent, self.max_parallel_paths, self.global_logger,
                        self.workspace_path, self.action_graph, self.registry, index=index,
                        global_info=self._new_info(index), policy=self.policy, max_step_num=self.max_step_num,
                        external_tools_enabled=self.external_tools_enabled, env=self.env, env_name=self.env_name,
                        audit=self.audit, path_uid=uid)
                    target.emit("path_created", inherited_steps=0, inherited_action_ids=[])
                new_paths.append(target)
            assignment = requested_assignments[slot] if slot < len(requested_assignments) else None
            target.global_info.task_signature = tuple(proposal.get("task_signature") or ())
            allocations.append(dict(path_uid=target.path_uid, action=action, action_id=action_id,
                                    assignment=assignment))
        self.reasoning_paths.extend(new_paths)
        if hasattr(self.policy, "accept"):
            self.policy.accept(proposal, info.path_uid, allocations)
        self.audit.emit("allocation", decision_id=decision_id, path_uid=info.path_uid,
                        accepted=allocations, rejected=requested[capacity:],
                        reason="reserved_capacity", capacity=capacity)
        by_uid = {p.path_uid: p for p in self.reasoning_paths}
        for allocation in allocations:
            target = by_uid[allocation["path_uid"]]
            if allocation["action"] == stop:
                self.audit.emit("action_finished", path_uid=target.path_uid,
                                action_id=allocation["action_id"], decision_id=decision_id,
                                status="stop", tokens=0, model_cost=0)
                probs = proposal.get("probs")
                target.finish("policy_stop", decision_id,
                              float(probs[0, -1].detach()) if probs is not None else None)
            else:
                target.reserve(self.registry.get_agent_from_idx(allocation["action"]),
                               allocation["action_id"], decision_id,
                               assignment=allocation.get("assignment"))

    def n_step(self, n):
        try:
            for _ in range(n):
                self.step()
                if self.check_finalize():
                    break
            if not self.check_finalize():
                raise RuntimeError("Runtime step budget exhausted with pending actions")
            return self.finalize()
        except BaseException as error:
            for path in self.reasoning_paths:
                path.finish("cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "execution_error")
            if hasattr(self.policy, "abort_task"):
                self.policy.abort_task()
            self.audit.emit("task_interrupted", error_type=type(error).__name__)
            raise

    def step(self):
        # Reserve/allocate immediately after each step; later paths see remaining capacity.
        for path in list(self.reasoning_paths):
            if path.stop_reason:
                continue
            path.step()
            if not path.stop_reason:
                self._route(path)
        self.update_graph()
        return self.answers

    def aggregate_answers(self, global_info, answers:list, query_func=None) -> str:
        if len(answers) == 0:
            main_logger.info("[Aggregation] skipped because this path has no answer candidates.")
            return None

        # A GAIA path is a stateful tool trajectory. Its last answer is expected
        # to incorporate the evidence collected earlier on that same path. An
        # answer-only LLM vote can discard that provenance and revive an obsolete
        # guess, so path selection is deliberately deterministic.
        if self.task.get("type") in {"GAIA", "MuSiQue"}:
            selected = select_gaia_path_answer(answers)
            main_logger.info("[Open QA Path Selection] last non-empty answer: %s", selected)
            return selected

        # only choose the last result without any format or extract
        if query_func is None:
            main_logger.info("[Aggregation] {}".format(answers[-1]))
            return answers[-1]

        # only choose the last result without any format or extract
        if self.task.get("type") == "SRDD" or self.task.get("type") == "CW":
            main_logger.info("[Aggregation] {}".format(global_info.code_path))
            return global_info.code_path

        prompt_filepath = "prompts/general/answer_prompt.json"
        with open(prompt_filepath, "r") as f:
            prompt = json.load(f)

        if self.task.get("type") == "MMLU" or self.task.get("type") == "MMLU-Pro":
            answer_prompt =  "\n".join(prompt["MMLU_aggregation"]).format(str(["{}\n".format(answer) for answer in answers]))
        elif self.task.get("type") == "GAIA":
            answer_prompt =  "\n".join(prompt["GAIA_aggregation"]).format(str(["{}\n".format(answer) for answer in answers]))
        elif self.task.get("type") == "GSM-Hard"  or self.task.get("type") == "gsm-hard" or self.task.get("type") == "GSM8K":
            answer_prompt = "\n".join(prompt["gsm_aggregation"]).format(str(["{}\n".format(answer) for answer in answers]))
        else:
            answer_prompt = "\n".join(prompt["answer_aggregation"]).format(str(["{}\n".format(answer) for answer in answers]))

        main_logger.info("[Aggregating] {}".format(answer_prompt))

        raw_response, _ = query_func(messages=answer_prompt)
        main_logger.info("[Aggregation Answer] {}".format(raw_response))

        task_type = self.task.get("type")
        if task_type in {"GSM-Hard", "gsm-hard", "GSM8K"}:
            # Base models can occasionally echo the aggregation prompt. In that case,
            # select from the actual candidate answers instead of scoring the echo.
            if (
                not raw_response
                or raw_response.strip() == answer_prompt.strip()
                or "You have several answer candidates." in raw_response
            ):
                raw_response = self.majority_vote(answers)

            number = BenchmarkEvaluator.extract_math_answer(raw_response)
            if number is None:
                raw_response = self.majority_vote(answers)
                number = BenchmarkEvaluator.extract_math_answer(raw_response)

            if number is None:
                return ""

            number = float(number)
            return str(int(number)) if number.is_integer() else str(number)

        return raw_response if raw_response else answers[-1]

    def majority_vote(self, answers: List) -> str:
        if self.task.get("type") == "MMLU" or self.task.get("type") == "MMLU-Pro":
            answers = [BenchmarkEvaluator.extract_choice_answer(answer) for answer in answers]
            main_logger.info("[Majority Vote] Answers: {}".format(answers))
        elif self.task.get("type") == "GSM-Hard" or self.task.get("type") == "gsm-hard" or self.task.get("type") == "GSM8K":
            answers = [BenchmarkEvaluator.extract_math_answer(answer) for answer in answers]
            main_logger.info("[Majority Vote] Answers: {}".format(answers))
        else:
            main_logger.info("[Majority Vote] Answers: {}".format(answers))

        answer_counts = {}
        for answer in answers:
            answer = str(answer).strip()  # Convert to string and remove whitespace
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

        if not answer_counts:
            return ""  # Return empty string if no answers

        max_count = max(answer_counts.values())
        most_common = [ans for ans, count in answer_counts.items() if count == max_count]
        main_logger.info("[Majority Vote] Most Common: {}".format(most_common))
        return most_common[-1]

    def finalize(self):
        candidates = []
        records = []
        from model.model_config import model_registry
        for path in self.reasoning_paths:
            info = path.global_info
            before = info.state_answers[-1] if info.state_answers else None
            agent = getattr(path, "last_agent", None)
            query_func = getattr(path, "last_query_func", None)
            with self.audit.call_scope(path_uid=path.path_uid, purpose="path_aggregation",
                                       model_size=(model_registry.get_model_size(agent.model) or 0) if agent else 0):
                prediction = self.aggregate_answers(info, info.state_answers, query_func)
            raw_prediction = prediction
            if self.task.get("type") in {"MMLU", "MMLU-Pro"} and self.runtime_config.get("aggregation", {}).get("mode", "legacy") != "legacy":
                prediction = normalize_choice(prediction, self.task.get("choices", "ABCDEFGHIJ")) or ""
            candidates.append(raw_prediction)
            record = dict(path_uid=path.path_uid, parent_path_uid=path.parent_path_uid,
                          before_aggregation=before, raw_prediction=raw_prediction,
                          prediction=prediction, stop_reason=path.stop_reason,
                          steps=[a.to_dict() for a in info.workflow.workflow])
            records.append(record)
            self.audit.emit("path_aggregation", **record)
        self.answers = [v for v in candidates if v is not None]
        aggregation_config = self.runtime_config.get("aggregation", {})
        selected_candidate_index = None
        if self.task.get("type") in {"MMLU", "MMLU-Pro"}:
            mode = aggregation_config.get("mode", "legacy")
            verifier = None
            if mode == "majority_verifier":
                model = aggregation_config.get("verifier_model")
                if not model:
                    raise ValueError("aggregation.verifier_model is required")
                from model.query_manager import query_manager
                def verifier(prompt):
                    with self.audit.call_scope(path_uid="task", purpose="tie_verifier",
                                               model_size=model_registry.get_model_size(model) or 0):
                        return query_manager.query(model, prompt)
            aggregation = aggregate_candidates(candidates, mode=mode,
                choices=self.task.get("choices", "ABCDEFGHIJ"),
                seed=int(aggregation_config.get("seed", 42)),
                task_id=str(self.task.get("id", "")), question=self.task.get("Question", ""), verifier=verifier)
            self.final_answer = aggregation["prediction"]
            if mode == "legacy" and len(self.answers) == 1:
                self.final_answer = self.answers[0]
                aggregation["prediction"] = self.final_answer
        elif self.task.get("type") == "GAIA":
            mode = aggregation_config.get("mode", "legacy")
            verifier = None
            if mode == "majority_verifier":
                model = aggregation_config.get("verifier_model")
                if not model:
                    raise ValueError("aggregation.verifier_model is required")
                from model.query_manager import query_manager

                def verifier(prompt):
                    with self.audit.call_scope(
                        path_uid="task",
                        purpose="gaia_answer_verifier",
                        model_size=model_registry.get_model_size(model) or 0,
                    ):
                        return query_manager.query(model, prompt)

            aggregation = aggregate_gaia_candidates(
                candidates,
                mode=mode,
                seed=int(aggregation_config.get("seed", 42)),
                task_id=str(self.task.get("id", "")),
                question=self.task.get("Question", ""),
                verifier=verifier,
            )
            self.final_answer = aggregation["prediction"]
        elif self.task.get("type") == "MuSiQue":
            mode = aggregation_config.get("mode", "legacy")
            verifier = None
            fallback_extractor = None
            routing_guard = getattr(self.policy, "routing_guard", None)
            if routing_guard is not None and routing_guard.llm_extraction_enabled:
                def fallback_extractor(candidate, index):
                    steps = records[index].get("steps") or []
                    role = steps[-1].get("agent") if steps else "path_aggregation"
                    parsed = routing_guard.parse_musique_result(
                        candidate,
                        question=self.task.get("Question", ""),
                        role=role,
                    )
                    status = (
                        "FINAL" if parsed.final_answer
                        else "CANDIDATE" if parsed.candidate_answer
                        else "NO_ANSWER"
                    )
                    return {
                        "status": status,
                        "answer": parsed.final_answer or parsed.candidate_answer,
                        "model": routing_guard.llm_extraction_model,
                        "tokens": 0,
                    }
            if mode == "majority_verifier":
                model = aggregation_config.get("verifier_model")
                if not model:
                    raise ValueError("aggregation.verifier_model is required")
                from model.query_manager import query_manager

                def verifier(prompt):
                    with self.audit.call_scope(
                        path_uid="task",
                        purpose="musique_answer_verifier",
                        model_size=model_registry.get_model_size(model) or 0,
                    ):
                        return query_manager.query(model, prompt)

            aggregation = aggregate_musique_candidates(
                candidates,
                mode=mode,
                seed=int(aggregation_config.get("seed", 42)),
                task_id=str(self.task.get("id", "")),
                question=self.task.get("Question", ""),
                verifier=verifier,
                fallback_extractor=fallback_extractor,
            )
            self.final_answer = aggregation["prediction"]
            selected_candidate_index = aggregation.get("selected_candidate_index")
        elif self.task.get("type") == "SRDD":
            aggregation = select_srdd_artifact(candidates)
            self.final_answer = aggregation["prediction"]
            selected_candidate_index = aggregation["selected_index"]
        elif len(self.answers) <= 1 or self.task.get("type") == "CW":
            self.final_answer = self.answers[-1] if self.answers else ""
            aggregation = dict(mode="last_artifact", prediction=self.final_answer)
        else:
            self.final_answer = self.majority_vote(self.answers)
            aggregation = dict(mode="legacy", prediction=self.final_answer)
        self.audit.emit("aggregation", **aggregation)
        snapshot = dict(**self.audit.context, split=self.runtime_config.get("audit_split", "unknown"),
                        question=self.task.get("Question", ""), choices=self.task.get("choices", "ABCDEFGHIJ"),
                        paths=records, candidates=candidates, candidate_hash=digest(candidates),
                        aggregation=aggregation, prediction=self.final_answer,
                        cost=self.audit.call_totals())
        self.audit.save("candidates.json", snapshot)
        self.audit.emit("prediction_committed", prediction=self.final_answer,
                        candidate_hash=snapshot["candidate_hash"])
        evaluations = []
        frozen_planner = getattr(self.policy, "routing_mode", "") == "frozen_llm_planner"
        final_success = None
        semantic_success = None
        effective_success = None
        semantic_evaluation = None
        final_metrics = {}
        final_reward = None
        musique_answer_scores = None
        if self.task.get("type") == "MuSiQue":
            musique_golds = [
                self.task.get("Answer"),
                *(self.task.get("answer_aliases") or ()),
            ]
            musique_answer_scores = BenchmarkEvaluator.musique_answer_scores(
                self.final_answer, musique_golds
            )
            musique_official_success = bool(musique_answer_scores["answer_em"])
            if bool(self._musique_semantic_config().get("enabled", False)):
                semantic_evaluation = self._judge_musique_post_task(
                    snapshot,
                    self.final_answer,
                    musique_golds,
                    official_success=musique_official_success,
                )
                semantic_success = bool(
                    semantic_evaluation.get("effective_equivalent")
                )
            else:
                semantic_success = musique_official_success
        for idx, (reasoning_path, aggregated_answer) in enumerate(zip(self.reasoning_paths, candidates)):
            if (frozen_planner and self.task.get("type") == "SRDD"
                    and idx != selected_candidate_index):
                evaluations.append(dict(path_uid=reasoning_path.path_uid, skipped=True,
                                        prediction=aggregated_answer))
                continue
            if self.task.get("type") in {"MMLU", "MMLU-Pro"} and aggregation_config.get("mode", "legacy") != "legacy":
                aggregated_answer = normalize_choice(aggregated_answer, self.task.get("choices", "ABCDEFGHIJ")) or ""
            if self.task.get("type") == "MMLU-Pro":
                correct = BenchmarkEvaluator.check_mmlu(
                    aggregated_answer, self.task.get("Answer")
                )
                transition = {
                'state': reasoning_path.global_info.workflow.state,
                'reward': self._binary_task_reward(correct),
                'action': None,
                'next_state': None,
                'done': True,
                'path_id': idx,
                'candidate_output': aggregated_answer,
                }
                print(transition)
                transition["path_uid"] = reasoning_path.path_uid
                self.policy.finalize_task(transition, reasoning_path.global_info)
            elif self.task.get("type") == "GSM-Hard":
                correct = BenchmarkEvaluator.check_gsm8k(
                    aggregated_answer, self.task.get("Answer")
                )
                transition = {
                'state': reasoning_path.global_info.workflow.state,
                'reward': self._binary_task_reward(correct),
                'action': None,
                'next_state': None,
                'done': True,
                'path_id': idx,
                'candidate_output': aggregated_answer,
                }
                print(transition)
                transition["path_uid"] = reasoning_path.path_uid
                self.policy.finalize_task(transition, reasoning_path.global_info)

            elif self.task.get("type") == "SRDD":
                reward, metrics = BenchmarkEvaluator.check_srdd(aggregated_answer, reasoning_path.global_info.task.get("Question"))
                transition = {
                'state': reasoning_path.global_info.workflow.state,
                'reward':  reward,
                'action': None,
                'next_state': None,
                'done': True,
                'path_id': idx,
                'candidate_output': aggregated_answer,
                "metrics":metrics
                }
                main_logger.info(metrics)
                transition["path_uid"] = reasoning_path.path_uid
                self.policy.finalize_task(transition, reasoning_path.global_info)

            elif self.task.get("type") == "GAIA":
                gold = self.task.get("Answer")
                correct = (
                    BenchmarkEvaluator.check_gaia(aggregated_answer, gold)
                    if gold is not None
                    else None
                )
                transition = {
                    'state': reasoning_path.global_info.workflow.state,
                    'reward': self._binary_task_reward(correct) if correct is not None else 0.0,
                    'action': None,
                    'next_state': None,
                    'done': True,
                    'path_id': idx,
                    'candidate_output': aggregated_answer,
                    'evaluable': gold is not None,
                }
                transition["path_uid"] = reasoning_path.path_uid
                self.policy.finalize_task(transition, reasoning_path.global_info)
            elif self.task.get("type") == "MuSiQue":
                golds = [
                    self.task.get("Answer"),
                    *(self.task.get("answer_aliases") or ()),
                ]
                answer_scores = BenchmarkEvaluator.musique_answer_scores(
                    aggregated_answer, golds
                )
                correct = bool(answer_scores["answer_em"])
                needs_training_signal = bool(
                    getattr(self.policy, "optimizer_updates_enabled", False)
                    and self._musique_semantic_enabled_for("training_reward")
                )
                needs_profile_signal = bool(
                    self.profile_evidence is not None
                    and self._musique_semantic_enabled_for("profile_evidence")
                )
                reward_success = (
                    bool(semantic_success)
                    if needs_training_signal and semantic_evaluation is not None
                    else correct
                )
                path_metrics = dict(answer_scores)
                if semantic_evaluation is not None:
                    path_metrics["task_semantic_evaluation"] = semantic_evaluation
                transition = {
                    'state': reasoning_path.global_info.workflow.state,
                    'reward': self._binary_task_reward(reward_success),
                    'action': None,
                    'next_state': None,
                    'done': True,
                    'path_id': idx,
                    'candidate_output': aggregated_answer,
                    'metrics': path_metrics,
                    'official_success': correct,
                    'semantic_success': (
                        bool(semantic_success)
                        if semantic_evaluation is not None
                        else None
                    ),
                    'profile_success': (
                        bool(semantic_success)
                        if needs_profile_signal and semantic_evaluation is not None
                        else correct
                    ),
                }
                transition["path_uid"] = reasoning_path.path_uid
                self.policy.finalize_task(transition, reasoning_path.global_info)
            elif self.task.get("type") == "CW":
                reward, metrics = BenchmarkEvaluator.check_commongen(concepts=reasoning_path.global_info.task.get("concepts"), text_path=aggregated_answer)
                transition = {
                'state': reasoning_path.global_info.workflow.state,
                'reward': reward,
                'action': None,
                'next_state': None,
                'done': True,
                'path_id': idx,
                'candidate_output': aggregated_answer,
                "metrics":metrics
                }
                main_logger.info(metrics)
                transition["path_uid"] = reasoning_path.path_uid
                self.policy.finalize_task(transition, reasoning_path.global_info)
            if self.profile_evidence is not None and transition.get("evaluable", True):
                task_type = self.task.get("type")
                if task_type == "SRDD":
                    profile_success = BenchmarkEvaluator.srdd_binary_success(
                        transition.get("metrics", {})
                    )
                elif task_type == "CW":
                    profile_success = BenchmarkEvaluator.commongen_binary_success(
                        transition.get("metrics", {})
                    )
                else:
                    profile_success = transition.get(
                        "profile_success", transition.get("reward", -1) > 0
                    )
                teammate_ids = tuple(
                    item.get("hash") for item in reasoning_path.agent_sequence
                    if item.get("hash") is not None
                )
                self.profile_evidence.record(
                    PathTerminalEvidence(
                        task_id=str(self.task.get("id")),
                        task_type=task_type,
                        path_id=idx,
                        teammate_ids=teammate_ids,
                        reward=float(profile_success),
                        role_adherence=reasoning_path.role_adherence,
                    )
                )

            evaluations.append(dict(
                path_uid=reasoning_path.path_uid,
                task_reward=transition["reward"],
                prediction=aggregated_answer,
                official_success=transition.get("official_success"),
                semantic_success=transition.get("semantic_success"),
            ))
        metrics = self.policy.update()
        if self.profile_evidence is not None:
            self.profile_evidence.flush_task(str(self.task.get("id")))
        for agent in self.registry.ordered_agents:
            agent.reset()
        if self.task.get("type") in {"MMLU", "MMLU-Pro"}:
            final_success = BenchmarkEvaluator.check_mmlu(
                self.final_answer, self.task.get("Answer")
            )
            final_reward = self._binary_task_reward(final_success)
        elif self.task.get("type") == "GAIA" and self.task.get("Answer") is not None:
            final_success = BenchmarkEvaluator.check_gaia(
                self.final_answer, self.task.get("Answer")
            )
            final_reward = self._binary_task_reward(final_success)
        elif self.task.get("type") == "MuSiQue":
            from role_aware.collaboration_metrics import (
                evaluate_musique_collaboration,
            )

            answer_scores = musique_answer_scores or BenchmarkEvaluator.musique_answer_scores(
                self.final_answer,
                [self.task.get("Answer"), *(self.task.get("answer_aliases") or ())],
            )
            collaboration = evaluate_musique_collaboration(
                records,
                self.task.get("gold_decomposition") or (),
                self.task.get("supporting_paragraph_indices") or (),
                self.task.get("Answer") or "",
                self.task.get("answer_aliases") or (),
                selected_candidate_index=selected_candidate_index,
            )
            final_success = bool(answer_scores["answer_em"])
            final_reward = self._binary_task_reward(final_success)
            if semantic_success is None:
                semantic_success = final_success
            effective_success = (
                semantic_success
                if self._musique_semantic_enabled_for("training_reward")
                else final_success
            )
            final_metrics = {
                **answer_scores,
                "official_success": final_success,
                "semantic_success": semantic_success,
                "semantic_evaluation": semantic_evaluation,
                "router_cs": (
                    {
                        key: semantic_evaluation.get(key)
                        for key in (
                            "prompt_version",
                            "routing_planning_score",
                            "state_handoff_communication_score",
                            "collaboration_score_raw",
                            "collaboration_score_100",
                            "planning_rationale",
                            "communication_rationale",
                            "failure_tags",
                            "critical_decision_ids",
                            "judge_model",
                            "judge_tokens",
                        )
                    }
                    if semantic_evaluation is not None
                    and semantic_evaluation.get("collaboration_evaluated")
                    else None
                ),
                "support_precision": collaboration["support_precision"],
                "support_recall": collaboration["support_recall"],
                "support_f1": collaboration["support_f1"],
                "paper_compatible_support_precision": collaboration[
                    "paper_compatible_support_precision"
                ],
                "paper_compatible_support_recall": collaboration[
                    "paper_compatible_support_recall"
                ],
                "paper_compatible_support_f1": collaboration[
                    "paper_compatible_support_f1"
                ],
                "paper_compatible_supporting_paragraphs": collaboration[
                    "paper_compatible_supporting_paragraphs"
                ],
                "paper_compatible_support_source": collaboration[
                    "paper_compatible_support_source"
                ],
                "collaboration": collaboration,
            }
        if effective_success is None:
            effective_success = final_success
        signatures = sorted({
            str(value)
            for path in self.reasoning_paths
            for value in (getattr(path.global_info, "task_signature", ()) or ())
            if str(value).strip()
        })
        routes = tuple(
            tuple(
                item.get("role") for item in path.agent_sequence
                if item.get("role") is not None
            )
            for path in self.reasoning_paths
        )
        experience_success = (
            semantic_success
            if (
                self.task.get("type") == "MuSiQue"
                and semantic_success is not None
                and self._musique_semantic_enabled_for("route_experience")
            )
            else final_success
        )
        self.experience_event = (
            RouteExperienceEvent(
                task_id=str(self.task.get("id")),
                dataset=str(self.task.get("type")),
                task_signature=tuple(signatures),
                routes=routes,
                success=bool(experience_success),
            )
            if experience_success is not None
            else None
        )
        self.final_outcome = {
            "success": final_success,
            "reward": final_reward,
            "semantic_success": semantic_success,
            "effective_success": effective_success,
            "experience_success": experience_success,
            "metrics": final_metrics,
        }
        self.audit.save("evaluation.json", dict(**self.audit.context, gold=self.task.get("Answer"),
                         answer_aliases=self.task.get("answer_aliases", []),
                         gold_decomposition=self.task.get("gold_decomposition", []),
                         supporting_paragraph_indices=self.task.get("supporting_paragraph_indices", []),
                         paths=evaluations, prediction=self.final_answer, policy_update=metrics,
                         final_outcome=self.final_outcome,
                         route_experience_event=(self.experience_event.to_dict()
                                                 if self.experience_event else None)))
        self.audit.emit("task_finished", prediction=self.final_answer, **self.audit.call_totals())
        main_logger.info("[Final Answer]: %s", self.final_answer)
        return self.final_answer, self.task.get("Answer")

    def visualize_path(self):
        for reasoning_path in self.reasoning_paths:
            reasoning_path.global_info.workflow.visualize()

    def visualize_graph(self):
        self.agent_graph.visualize(os.path.join(self.workspace_path, "agent_graph.html"))
        self.action_graph.visualize(os.path.join(self.workspace_path, "action_graph.html"))

    def print_paths(self):
        for reasoning_path in self.reasoning_paths:
            main_logger.info("Reasoning Path: {}\nAgent Sequence: {}\n".format(reasoning_path.index, reasoning_path.print_agent_sequence()))

    def format_index(self):
        for index, reasoning_path in enumerate(self.reasoning_paths):
            reasoning_path.index = index

    def update_graph(self):
        for index, reasoning_path in enumerate(self.reasoning_paths):
            for successor, predecessor in zip(reasoning_path.agent_sequence[:-1], reasoning_path.agent_sequence[1:]):
                successor = self.registry.get_agent_from_idx(successor.get("hash"))
                predecessor = self.registry.get_agent_from_idx(predecessor.get("hash"))
                res = self.agent_graph._get_edge(predecessor, successor)
                if res is None or index not in res:
                    self.agent_graph._add_edge(predecessor, successor, index)

    def check_finalize(self):
        for reasoning_path in self.reasoning_paths[:self.max_parallel_paths]:
            if reasoning_path.state != ReasoningState.FINALIZING and reasoning_path.state != ReasoningState.DISCARDING:
                return False
        return True
