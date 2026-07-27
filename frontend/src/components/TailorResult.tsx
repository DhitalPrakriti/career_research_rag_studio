import { useState } from "react";
import type { Requirement, RequirementStatus, TailorResponse } from "../api";
import { Badge } from "./Badge";

const STATUS_TONE: Record<RequirementStatus, "good" | "warning" | "critical"> = {
  matched: "good",
  partial: "warning",
  missing: "critical",
};

const STATUS_LABEL: Record<RequirementStatus, string> = {
  matched: "Matched",
  partial: "Partial",
  missing: "Not in your resume",
};

function RequirementRow({
  requirement,
  bulletText,
}: {
  requirement: Requirement;
  bulletText: string | undefined;
}) {
  return (
    <details className="requirement" open={requirement.status === "missing"}>
      <summary>
        <Badge tone={STATUS_TONE[requirement.status]}>
          {STATUS_LABEL[requirement.status]}
        </Badge>
        <span className="requirement-text">{requirement.text}</span>
      </summary>

      {requirement.status === "missing" ? (
        <p className="requirement-note">
          Nothing in your resume supports this, so no bullet was written for it. Add real
          evidence if you have it elsewhere — the tool will not invent it.
        </p>
      ) : (
        <>
          {bulletText && (
            <p className="requirement-bullet">
              <span aria-hidden="true">▸ </span>
              {bulletText}
            </p>
          )}
          <p className="requirement-note">Written from your own resume text:</p>
          <ul className="evidence">
            {requirement.evidence.map((item) => (
              <li key={`${requirement.id}-${item.source_id}-${item.title}`}>
                <span className="source-id">[{item.source_id}]</span> {item.text}
              </li>
            ))}
          </ul>
        </>
      )}
    </details>
  );
}

export function TailorResult({ result }: { result: TailorResponse }) {
  const [copied, setCopied] = useState(false);
  const coveragePercent = Math.round(result.coverage * 100);
  const bulletByRequirement = new Map(
    result.bullets.map((bullet) => [bullet.requirement_id, bullet.text]),
  );

  const copyBullets = async () => {
    const text = result.bullets.map((bullet) => `• ${bullet.text}`).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <>
      <dl className="stats">
        <div className="stat">
          <dt>Coverage</dt>
          <dd>
            {coveragePercent}
            <span className="unit">%</span>
          </dd>
          <div
            className="meter"
            role="img"
            aria-label={`Requirement coverage ${coveragePercent} percent`}
          >
            <i style={{ width: `${Math.max(coveragePercent, 2)}%` }} />
          </div>
        </div>
        <div className="stat">
          <dt>Matched</dt>
          <dd>{result.matched_count}</dd>
        </div>
        <div className="stat">
          <dt>Partial</dt>
          <dd>{result.partial_count}</dd>
        </div>
        <div className="stat">
          <dt>Gaps</dt>
          <dd>{result.missing_count}</dd>
        </div>
      </dl>

      {result.recommended_resume && (
        <section className="card" style={{ marginTop: 16 }}>
          <h2>Best starting point</h2>
          <p className="answer-body">
            <strong>{result.recommended_resume}</strong> supplied the most supporting
            evidence for this job, so start from that version.
          </p>
        </section>
      )}

      <section className="card">
        <div className="answer-head">
          <h2 style={{ margin: 0, flex: 1 }}>Tailored bullets</h2>
          {result.bullets.length > 0 && (
            <button type="button" className="chip" onClick={copyBullets}>
              {copied ? "Copied" : "Copy all"}
            </button>
          )}
        </div>

        {result.bullets.length === 0 ? (
          <p className="empty">
            No bullets were generated.{" "}
            {result.is_generated
              ? "None of the requirements had supporting evidence."
              : "Configure an LLM provider in .env to generate them — the gap analysis below is still valid."}
          </p>
        ) : (
          <ul className="bullets">
            {result.bullets.map((bullet) => (
              <li key={bullet.requirement_id}>
                {bullet.text}
                <span className="bullet-sources">
                  {bullet.source_ids.map((id) => `[${id}]`).join(" ")}
                </span>
              </li>
            ))}
          </ul>
        )}

        <p className="notice">
          Every bullet is rewritten from text already in your resume. Requirements with no
          evidence are listed as gaps and never written about — so this cannot claim
          experience you do not have.
        </p>
      </section>

      <section className="card">
        <h2>Requirement by requirement</h2>
        {result.requirements.length === 0 ? (
          <p className="empty">No requirements were extracted from that job description.</p>
        ) : (
          result.requirements.map((requirement) => (
            <RequirementRow
              key={requirement.id}
              requirement={requirement}
              bulletText={bulletByRequirement.get(requirement.id)}
            />
          ))
        )}
      </section>
    </>
  );
}
