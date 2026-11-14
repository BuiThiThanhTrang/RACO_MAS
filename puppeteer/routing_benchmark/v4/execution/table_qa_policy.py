"""Reference policy for calibrating table QA packets with the live runner."""
from routing_benchmark.routers import RoutingDecision


class TableQAFixedPolicy:
    name = "table_qa_fixed"

    def reset(self, seed=None):
        self.previous = None
        self.recovery = []

    def select(self, observation, candidate_actions, oracle_valid_actions=None):
        state = observation["current_state"]
        current, artifacts = state["current"], state["artifacts"]
        fresh = {kind for kind, ref in current.items() if artifacts[ref]["fresh"]}
        if state["terminal"]:
            action = "STOP"
        elif self.recovery:
            action = self.recovery.pop(0)
            if action == "Answer Composer" and "answer" in fresh:
                calculation = current.get("calculation")
                if calculation in artifacts[current["answer"]]["depends_on"]:
                    action = "Evidence Verifier"
        elif self.previous is None:
            action = "Table Inspector"
        elif any(table["hierarchical"] for table in state["tables"]) and "headers" not in fresh:
            action = "Header Resolver"
        elif "verification" in fresh and not artifacts[current["verification"]]["data"]["passed"]:
            if self.previous == "Failure Diagnoser" and "diagnosis" in fresh:
                step = artifacts[current["diagnosis"]]["data"]["step"]
                action, *self.recovery = {
                    "headers": ["Header Resolver", "Evidence Selector", "Numerical Analyst", "Answer Composer"],
                    "selection": ["Evidence Selector", "Numerical Analyst", "Answer Composer"],
                    "calculation": ["Numerical Analyst", "Answer Composer"],
                    "answer": ["Answer Composer"],
                }[step]
            else:
                action = "Failure Diagnoser"
        elif "answer" in fresh:
            action = "Evidence Verifier"
        elif "selection" not in fresh and self.previous != "Evidence Selector":
            action = "Evidence Selector"
        elif "calculation" not in fresh and self.previous != "Numerical Analyst":
            action = "Numerical Analyst"
        else:
            action = "Answer Composer"
        self.previous = action
        return RoutingDecision(action)


class TableQAPredefinedPolicy(TableQAFixedPolicy):
    """Deterministic specialist trajectory driven only by public state."""

    name = "table_qa_predefined"


class TableQASingleAgentPolicy:
    """Dispatch one generalist until verification passes, then stop."""

    name = "table_qa_single_agent"
    agent_id = "Table QA Agent"

    def reset(self, seed=None):
        del seed

    def select(self, observation, candidate_actions, oracle_valid_actions=None):
        del oracle_valid_actions
        action = "STOP" if observation["current_state"]["terminal"] else self.agent_id
        if action not in candidate_actions:
            raise ValueError(f"Single-agent action is unavailable: {action}")
        return RoutingDecision(action)
