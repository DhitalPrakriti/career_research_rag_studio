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


class TailorRequest(BaseModel):
    job_description: str = Field(min_length=20, max_length=20000)
    max_requirements: int = Field(default=25, ge=1, le=40)


class RequirementOut(BaseModel):
    id: int
    text: str
    status: str = Field(description="matched, partial or missing")
    score: float
    evidence: list[ContextOut]


class BulletOut(BaseModel):
    requirement_id: int
    text: str
    source_ids: list[int]


class TailorResponse(BaseModel):
    provider: str
    is_generated: bool = Field(
        description=(
            "False when no bullets were generated — no LLM configured, or the rewrite "
            "could not be parsed. The gap analysis is still valid."
        )
    )
    coverage: float = Field(description="Share of requirements with evidence; partial counts half.")
    recommended_resume: str | None
    matched_count: int
    partial_count: int
    missing_count: int
    requirements: list[RequirementOut]
    bullets: list[BulletOut]
    citations: list[CitationOut]
    contexts: list[ContextOut]
    trace: list[TraceEventOut]
    elapsed_ms: int


class DocumentOut(BaseModel):
    title: str
    pages: int | None = None
    chunks: int = 0


class StoredDocumentOut(BaseModel):
    name: str
    size_bytes: int
    modified: str
    chunks: int = 0


class DocumentsResponse(BaseModel):
    documents: list[StoredDocumentOut]
    chunks: int
    writes_enabled: bool = Field(
        description=(
            "False when ALLOW_DOCUMENT_WRITES is off. These endpoints have no auth, so a "
            "public deployment should disable them."
        )
    )
    allowed_types: list[str]
    max_upload_bytes: int


class ReindexResponse(BaseModel):
    documents: list[StoredDocumentOut]
    chunks: int
    elapsed_ms: int
    message: str


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=200)


class SessionResponse(BaseModel):
    auth_required: bool = Field(
        description="False when APP_PASSWORD is unset, which leaves the API open."
    )
    authenticated: bool


class HealthResponse(BaseModel):
    status: str
    provider: str
    is_generated: bool
    documents: list[DocumentOut]
    chunks: int
    writes_enabled: bool = True
