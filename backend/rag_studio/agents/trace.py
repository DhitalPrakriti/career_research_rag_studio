from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentTraceEvent:
    node: str
    message: str
    details: dict[str, Any]


def add_trace_event(
    trace: list[AgentTraceEvent],
    node: str,
    message: str,
    **details: Any,
) -> list[AgentTraceEvent]:
    return [
        *trace,
        AgentTraceEvent(
            node=node,
            message=message,
            details=details,
        ),
    ]


def format_trace(trace: list[AgentTraceEvent]) -> str:
    if not trace:
        return "Trace: no events recorded."

    lines = ["Trace"]
    for index, event in enumerate(trace, start=1):
        lines.append(f"{index}. {event.node}: {event.message}")
        for key, value in event.details.items():
            lines.append(f"   {key}: {value}")
    return "\n".join(lines)
