"""Request and response shapes for the API.

Pydantic models rather than the internal dataclasses: the wire format should be able
to change independently of the retrieval internals, and FastAPI generates the OpenAPI
schema from these.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    source_id: int
    title: str
    location: str
    score: float


class ContextOut(BaseModel):
    source_id: int
    text: str
    score: float
    title: str
    page: int | None = None
    chunk_index: int | None = None


class TraceEventOut(BaseModel):
    step: int
    node: str
    message: str
    details: dict[str, str]


class RouteOut(BaseModel):
    route: str
    retriever: str
    parent_context: bool
    multi_query: bool
    hyde: bool
    rewrite_before_retrieval: bool
    reason: str


class GradeOut(BaseModel):
    is_relevant: bool
    score: float
    reason: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    provider: str
    is_generated: bool = Field(
        description=(
            "False when the answer is extractive fallback text rather than LLM output. "
            "The UI surfaces this so a keyless run is never mistaken for generation."
        )
    )
    refused: bool = Field(
        description="Whether the answer states the information is not in the sources."
    )
    route: RouteOut | None = None
    grade: GradeOut | None = None
    retry_count: int
    citations: list[CitationOut]
    contexts: list[ContextOut]
    trace: list[TraceEventOut]
    elapsed_ms: int


class DocumentOut(BaseModel):
    title: str
    pages: int | None = None


class HealthResponse(BaseModel):
    status: str
    provider: str
    is_generated: bool
    documents: list[DocumentOut]
    chunks: int
