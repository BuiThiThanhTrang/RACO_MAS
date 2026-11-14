from __future__ import annotations

from typing import Any


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _short(values: list[str], limit: int = 3) -> str:
    if not values:
        return ""
    shown = values[:limit]
    suffix = f" +{len(values) - limit}" if len(values) > limit else ""
    return ", ".join(shown) + suffix


def workflow_to_dot(
    scenario: dict[str, Any], agent_names: dict[str, str] | None = None
) -> str:
    """Render formal transitions as a dependency graph in Graphviz DOT."""
    agent_names = agent_names or {}
    transitions = scenario.get("transitions", [])
    initial_facts = set(scenario.get("initial_facts", []))
    initial_artifacts = set(scenario.get("initial_artifacts", []))
    goal_facts = set(scenario.get("goal_facts", []))
    goal_artifacts = set(scenario.get("goal_artifacts", []))

    lines = [
        "digraph workflow {",
        '  graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.55"];',
        '  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, color="#CBD5E1"];',
        '  edge [fontname="Arial", fontsize=8, color="#64748B"];',
        '  start [shape=circle, label="START", fillcolor="#DBEAFE", color="#3B82F6"];',
        '  goal [shape=doublecircle, label="GOAL", fillcolor="#DCFCE7", color="#22C55E"];',
    ]

    produced_by: dict[str, list[int]] = {}
    required_by: dict[str, list[int]] = {}
    for index, transition in enumerate(transitions):
        action = str(transition.get("action", "unknown"))
        role = agent_names.get(action, action)
        effect = str(transition.get("public_effect", ""))
        status = str(transition.get("status", ""))
        color = "#FEF3C7" if "FAIL" in status or "REJECT" in status else "#F8FAFC"
        label = f"{role}\\n{effect}" if effect else role
        lines.append(
            f'  t{index} [label="{_escape(label)}", fillcolor="{color}", tooltip="{_escape(transition.get("transition_id", ""))}"];'
        )
        requirements = list(transition.get("requires_all", [])) + list(
            transition.get("requires_artifacts", [])
        )
        requirements += list(transition.get("requires_any", []))
        products = list(transition.get("add_facts", [])) + list(
            transition.get("add_artifacts", [])
        )
        for item in requirements:
            required_by.setdefault(str(item), []).append(index)
        for item in products:
            produced_by.setdefault(str(item), []).append(index)

        if set(requirements).issubset(initial_facts | initial_artifacts):
            label = _short([str(item) for item in requirements]) or "initial"
            lines.append(f'  start -> t{index} [label="{_escape(label)}"];')
        if goal_facts.intersection(products) or goal_artifacts.intersection(products):
            label = _short(
                sorted((goal_facts | goal_artifacts).intersection(products))
            )
            lines.append(f'  t{index} -> goal [label="{_escape(label)}"];')

    seen_edges: set[tuple[int, int, str]] = set()
    for artifact, producers in produced_by.items():
        for producer in producers:
            for consumer in required_by.get(artifact, []):
                if producer == consumer:
                    continue
                edge = (producer, consumer, artifact)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                lines.append(
                    f'  t{producer} -> t{consumer} [label="{_escape(artifact)}"];'
                )

    if not transitions:
        lines.append('  start -> goal [style=dashed, label="no transitions"];')
    lines.append("}")
    return "\n".join(lines)
