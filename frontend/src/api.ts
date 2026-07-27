// Types mirror backend/rag_studio/api/models.py. They are hand-written rather than
// generated so the shape is reviewable in one place; if they drift, the OpenAPI schema
// at /docs is the source of truth.

export interface Citation {
  source_id: number;
  title: string;
  location: string;
  score: number;
}

export interface Context {
  source_id: number;
  text: string;
  score: number;
  title: string;
  page: number | null;
  chunk_index: number | null;
}

export interface TraceEvent {
  step: number;
  node: string;
  message: string;
  details: Record<string, string>;
}

export interface Route {
  route: string;
  retriever: string;
  parent_context: boolean;
  multi_query: boolean;
  hyde: boolean;
  rewrite_before_retrieval: boolean;
  reason: string;
}

export interface Grade {
  is_relevant: boolean;
  score: number;
  reason: string;
}

export interface QueryResponse {
  question: string;
  answer: string;
  provider: string;
  is_generated: boolean;
  refused: boolean;
  route: Route | null;
  grade: Grade | null;
  retry_count: number;
  citations: Citation[];
  contexts: Context[];
  trace: TraceEvent[];
  elapsed_ms: number;
}

export type RequirementStatus = "matched" | "partial" | "missing";

export interface Requirement {
  id: number;
  text: string;
  status: RequirementStatus;
  score: number;
  evidence: Context[];
}

export interface Bullet {
  requirement_id: number;
  text: string;
  source_ids: number[];
}

export interface TailorResponse {
  provider: string;
  is_generated: boolean;
  coverage: number;
  recommended_resume: string | null;
  matched_count: number;
  partial_count: number;
  missing_count: number;
  requirements: Requirement[];
  bullets: Bullet[];
  citations: Citation[];
  contexts: Context[];
  trace: TraceEvent[];
  elapsed_ms: number;
}

export interface HealthResponse {
  status: string;
  provider: string;
  is_generated: boolean;
  documents: { title: string; pages: number | null; chunks: number }[];
  chunks: number;
  writes_enabled: boolean;
}

export interface StoredDocument {
  name: string;
  size_bytes: number;
  modified: string;
  chunks: number;
}

export interface DocumentsResponse {
  documents: StoredDocument[];
  chunks: number;
  writes_enabled: boolean;
  allowed_types: string[];
  max_upload_bytes: number;
}

export interface ReindexResponse {
  documents: StoredDocument[];
  chunks: number;
  elapsed_ms: number;
  message: string;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return "Invalid request.";
  } catch {
    // fall through to the status line
  }
  return `Request failed (${response.status} ${response.statusText}).`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as HealthResponse;
}

export async function askQuestion(
  question: string,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as QueryResponse;
}

export async function fetchDocuments(): Promise<DocumentsResponse> {
  const response = await fetch("/api/documents");
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as DocumentsResponse;
}

export async function uploadDocuments(files: FileList | File[]): Promise<ReindexResponse> {
  const body = new FormData();
  for (const file of Array.from(files)) body.append("files", file);

  const response = await fetch("/api/documents", { method: "POST", body });
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as ReindexResponse;
}

export async function reindexDocuments(): Promise<ReindexResponse> {
  const response = await fetch("/api/documents/reindex", { method: "POST" });
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as ReindexResponse;
}

export async function deleteDocument(name: string): Promise<ReindexResponse> {
  const response = await fetch(`/api/documents/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as ReindexResponse;
}

export async function tailorResume(
  jobDescription: string,
  signal?: AbortSignal,
): Promise<TailorResponse> {
  const response = await fetch("/api/tailor", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_description: jobDescription }),
    signal,
  });
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as TailorResponse;
}
