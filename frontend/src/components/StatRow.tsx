import type { QueryResponse } from "../api";

interface StatRowProps {
  result: QueryResponse;
}

/** Bare stat tiles: single headline numbers, so no plot and no hover layer.
 *  The grade tile adds a one-hue sequential meter because 0–1 has a fixed domain. */
export function StatRow({ result }: StatRowProps) {
  const grade = result.grade;
  const gradePercent = grade ? Math.round(grade.score * 100) : null;

  return (
    <dl className="stats">
      <div className="stat">
        <dt>Latency</dt>
        <dd>
          {(result.elapsed_ms / 1000).toFixed(1)}
          <span className="unit">s</span>
        </dd>
      </div>

      <div className="stat">
        <dt>Retrieval grade</dt>
        <dd>
          {gradePercent === null ? "—" : gradePercent}
          {gradePercent !== null && <span className="unit">%</span>}
        </dd>
        {gradePercent !== null && (
          <div
            className="meter"
            role="img"
            aria-label={`Retrieval grade ${gradePercent} percent`}
          >
            <i style={{ width: `${Math.max(gradePercent, 2)}%` }} />
          </div>
        )}
      </div>

      <div className="stat">
        <dt>Sources cited</dt>
        <dd>{result.citations.length}</dd>
      </div>

      <div className="stat">
        <dt>Rewrite retries</dt>
        <dd>{result.retry_count}</dd>
      </div>
    </dl>
  );
}
