import { useEffect, useRef, useState } from "react";
import {
  deleteDocument,
  fetchDocuments,
  reindexDocuments,
  uploadDocuments,
  type DocumentsResponse,
} from "../api";
import { Badge } from "./Badge";

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatModified(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

export function DocumentsPanel({ onChanged }: { onChanged: () => void }) {
  const [data, setData] = useState<DocumentsResponse | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = () => {
    fetchDocuments()
      .then(setData)
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Failed to load."));
  };

  useEffect(load, []);

  const act = async (label: string, task: () => Promise<{ message: string }>) => {
    setBusy(label);
    setError(null);
    setMessage(null);
    try {
      const result = await task();
      setMessage(result.message);
      load();
      onChanged();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      setBusy(null);
      setConfirming(null);
    }
  };

  const writesOff = data !== null && !data.writes_enabled;
  const staleCount = data?.documents.filter((document) => document.chunks === 0).length ?? 0;

  return (
    <section className="card" style={{ marginTop: 16 }}>
      <div className="answer-head">
        <h2 style={{ margin: 0, flex: 1 }}>My documents</h2>
        {data && <Badge tone="neutral">{data.chunks} chunks indexed</Badge>}
      </div>

      {writesOff && (
        <p className="notice">
          Document changes are disabled on this server (<code>ALLOW_DOCUMENT_WRITES</code>).
          You can still see what is indexed.
        </p>
      )}

      {data && data.documents.length === 0 ? (
        <p className="empty">
          No documents yet. Upload a resume PDF to get started.
        </p>
      ) : (
        <ul className="documents">
          {data?.documents.map((document) => (
            <li key={document.name}>
              <div className="document-main">
                <span className="document-name">{document.name}</span>
                <span className="document-meta">
                  {formatSize(document.size_bytes)} · updated {formatModified(document.modified)}
                </span>
              </div>
              {document.chunks === 0 ? (
                <Badge tone="warning">Not indexed</Badge>
              ) : (
                <Badge tone="good">{document.chunks} chunks</Badge>
              )}
              {!writesOff &&
                (confirming === document.name ? (
                  <span className="confirm">
                    <button
                      type="button"
                      className="chip is-danger"
                      disabled={busy !== null}
                      onClick={() => void act("delete", () => deleteDocument(document.name))}
                    >
                      Really remove
                    </button>
                    <button
                      type="button"
                      className="chip"
                      onClick={() => setConfirming(null)}
                    >
                      Cancel
                    </button>
                  </span>
                ) : (
                  <button
                    type="button"
                    className="chip"
                    disabled={busy !== null}
                    onClick={() => setConfirming(document.name)}
                  >
                    Remove
                  </button>
                ))}
            </li>
          ))}
        </ul>
      )}

      {staleCount > 0 && (
        <p className="notice">
          {staleCount} file{staleCount === 1 ? " is" : "s are"} on disk but not in the index.
          Reindex to include {staleCount === 1 ? "it" : "them"}.
        </p>
      )}

      {!writesOff && (
        <div className="form-actions" style={{ marginTop: 14 }}>
          <input
            ref={fileInput}
            type="file"
            multiple
            accept={data?.allowed_types.join(",") ?? ".pdf,.txt,.md"}
            style={{ display: "none" }}
            onChange={(event) => {
              const files = event.target.files;
              if (files && files.length > 0) {
                void act("upload", () => uploadDocuments(files));
              }
              event.target.value = "";
            }}
          />
          <button
            type="button"
            className="chip"
            disabled={busy !== null}
            onClick={() => fileInput.current?.click()}
          >
            {busy === "upload" ? "Uploading…" : "Upload resume"}
          </button>
          <button
            type="button"
            className="chip"
            disabled={busy !== null}
            onClick={() => void act("reindex", reindexDocuments)}
          >
            {busy === "reindex" ? "Reindexing…" : "Reindex"}
          </button>
          <span className="document-hint">
            Uploading a file with the same name replaces it, then reindexes automatically.
          </span>
        </div>
      )}

      {busy && (
        <p className="notice">
          <span className="spinner" aria-hidden="true" />
          Rebuilding the index — this takes a few seconds.
        </p>
      )}

      {message && !busy && <p className="notice">{message}</p>}

      {error && (
        <p className="notice is-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
