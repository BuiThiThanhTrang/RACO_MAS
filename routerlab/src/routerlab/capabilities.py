from __future__ import annotations

from .contracts import ROUTER_ACTIONS


# Canonical, router-facing descriptions of the action transition semantics.
# These are deliberately separate from executor role prompts: routers need to
# know what each action can observe and change, but not output-format details.
ROUTER_CAPABILITY_CARDS: dict[str, str] = {
    "coder": (
        "Inputs: task, optional current code, Base feedback, and prior findings. "
        "Effect: writes or replaces code; changed code is automatically Base-tested. "
        "Availability: with or without current code, for any Base outcome. Cost: 1 step."
    ),
    "logic_reviewer": (
        "Inputs: task, optional current code, and Base feedback. Effect: produces "
        "logical-defect findings for later code-producing agents; does not change code "
        "or run tests. Availability: with or without current code, whether Base Tests "
        "pass or fail. Cost: 1 step."
    ),
    "edge_case_analyst": (
        "Inputs: task, optional current code, and Base feedback. Effect: produces "
        "boundary and input-space findings for later code-producing agents; does not "
        "change code or run tests. Availability: with or without current code, whether "
        "Base Tests pass or fail. Cost: 1 step."
    ),
    "repair_agent": (
        "Inputs: current code, Base feedback, and prior findings. Effect: repairs "
        "existing code; changed code is automatically Base-tested. Availability: "
        "requires current code; otherwise this action is a charged no-op. Cost: 1 step."
    ),
    "STOP": (
        "Inputs: current state. Effect: ends the episode and submits the current code; "
        "with no current code, ends without a submission and the task fails. Cost: 0 steps."
    ),
}


if tuple(ROUTER_CAPABILITY_CARDS) != ROUTER_ACTIONS:
    raise RuntimeError("Router capability cards must exactly match ROUTER_ACTIONS order")


def router_capability_payload() -> dict[str, str]:
    """Return a copy safe to place in a provider request payload."""

    return dict(ROUTER_CAPABILITY_CARDS)
