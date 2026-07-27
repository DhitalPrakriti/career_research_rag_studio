from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_LANGSMITH_PROJECT = "career-research-rag-studio"


@dataclass(frozen=True)
class LangSmithSettings:
    enabled: bool
    project: str


def configure_langsmith(
    enabled: bool,
    project: str | None = None,
) -> LangSmithSettings:
    project_name = project or os.getenv("LANGSMITH_PROJECT") or DEFAULT_LANGSMITH_PROJECT
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        return LangSmithSettings(enabled=False, project=project_name)

    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError(
            "LANGSMITH_API_KEY is required when --langsmith is enabled. "
            "Create an API key in LangSmith, then set it in your terminal or .env file."
        )

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project_name
    return LangSmithSettings(enabled=True, project=project_name)


def langsmith_run_config(
    settings: LangSmithSettings,
    run_name: str,
) -> dict[str, object] | None:
    if not settings.enabled:
        return None
    return {
        "run_name": run_name,
        "tags": ["career-rag-studio", "langgraph-agent"],
        "metadata": {
            "project": settings.project,
            "component": "CareerResearchAgent",
        },
    }
