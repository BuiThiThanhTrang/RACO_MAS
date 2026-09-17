from typing import List
from pathlib import Path
from contextlib import nullcontext
from role_aware.audit_trace import AuditTrace, digest, redact
from role_aware.aggregation import aggregate_candidates, normalize_choice
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
        public_task = {k: v for k, v in self.task.items() if k not in {"Answer", "answer", "target"}}
        info = GlobalInfo(path_id=index, workpath=self.workspace_path, task=public_task,
                          env=self.env, env_name=self.env_name)
        info.audit_split = self.runtime_config.get("audit_split", "unknown")
        return info

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
            allocations.append(dict(path_uid=target.path_uid, action=action, action_id=action_id))
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
                               allocation["action_id"], decision_id)

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
        elif len(self.answers) <= 1 or self.task.get("type") in {"CW", "SRDD"}:
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
        for idx, (reasoning_path, aggregated_answer) in enumerate(zip(self.reasoning_paths, candidates)):
            if self.task.get("type") in {"MMLU", "MMLU-Pro"} and aggregation_config.get("mode", "legacy") != "legacy":
                aggregated_answer = normalize_choice(aggregated_answer, self.task.get("choices", "ABCDEFGHIJ")) or ""
            if self.task.get("type") == "MMLU-Pro":
                transition = {
                'state': reasoning_path.global_info.workflow.state,
                'reward': 1 if BenchmarkEvaluator.check_mmlu(aggregated_answer, self.task.get("Answer")) else -1,
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
                transition = {
                'state': reasoning_path.global_info.workflow.state,
                'reward': 1 if BenchmarkEvaluator.check_gsm8k(aggregated_answer, self.task.get("Answer")) else -1,
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
            if self.profile_evidence is not None:
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
                    profile_success = transition.get("reward", -1) > 0
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

            evaluations.append(dict(path_uid=reasoning_path.path_uid, task_reward=transition["reward"],
                                    prediction=aggregated_answer))
        metrics = self.policy.update()
        if self.profile_evidence is not None:
            self.profile_evidence.flush_task(str(self.task.get("id")))
        for agent in self.registry.ordered_agents:
            agent.reset()
        self.audit.save("evaluation.json", dict(**self.audit.context, gold=self.task.get("Answer"),
                         paths=evaluations, prediction=self.final_answer, policy_update=metrics))
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
