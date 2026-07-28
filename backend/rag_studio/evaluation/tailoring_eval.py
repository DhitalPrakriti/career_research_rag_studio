"""Evaluate the tailoring pipeline against labelled job descriptions.

The unit tests cover the mechanics — a gap cannot produce a bullet, thresholds classify,
malformed model output is refused. They say nothing about whether the output is *right* on
a real posting, which is what this measures.

Three metrics, in descending order of how much they matter:

1. `fabrication_rate` — the share of generated bullets containing a checkable claim that
   does not appear in the evidence the bullet was written from. This needs no hand
   labelling and independently tests the guarantee the whole design rests on. A structural
   argument that gaps cannot become bullets is not the same as evidence that the bullets
   which *are* produced stay inside their evidence.
2. `classification_accuracy` — how often a requirement's status agrees with a human label.
   This is what would have caught "Kubernetes matched against a resume with no Kubernetes"
   automatically, rather than by happening to read the output.
3. `extraction_recall` — whether the requirements a human considers assessable were found
   at all. A requirement that is never extracted can never be scored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_studio.tailoring.matching import MATCHED, MISSING, PARTIAL

# A claim is only worth checking if a wrong one would be a real misrepresentation and a
# false alarm is unlikely. Numbers and technology-shaped tokens qualify; ordinary English
# does not, so plain capitalised words are ignored entirely.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_TECH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./-]*")

# Tokens that look technical but are ordinary writing, or are so generic that presence in
# the evidence is not meaningful. The second block matters because the named-entity rule
# fires on any mid-sentence capital: without it, a bullet reading "...with the Team on the
# Project" reports two fabrications and the real signal drowns in noise.
_CLAIM_STOP_WORDS = frozenset(
    """
    a an and the of to for with in on at by from as is are was were be been being
    built designed developed created implemented deployed achieved evaluated applied
    utilized utilised used leveraged conducted executed integrated delivered led
    including such that this these those it its their they i my our we
    end-to-end full-stack real-world production-oriented state-of-the-art

    team project company client customer stakeholder manager management
    data system systems service services platform cloud software application
    engineer engineering developer development experience role position
    pipeline pipelines model models architecture architectures framework frameworks
    api apis database databases server servers testing evaluation deployment
    """.split()
)


@dataclass(frozen=True)
class TailoringExample:
    """One labelled posting.

    Labels are keywords rather than full requirement text, because requirement wording is
    produced by an LLM and paraphrases freely — "Experience with FAISS or comparable vector
    databases" and "Familiarity with FAISS" are the same requirement. Matching on the
    keyword survives that; matching on exact text would not.
    """

    id: str
    job_description: str
    present: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    notes: str = ""


def load_tailoring_set(path: str | Path) -> list[TailoringExample]:
    examples: list[TailoringExample] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            raw = json.loads(line)
            examples.append(
                TailoringExample(
                    id=str(raw["id"]),
                    job_description=str(raw["job_description"]),
                    present=[str(item) for item in raw.get("present", [])],
                    absent=[str(item) for item in raw.get("absent", [])],
                    notes=str(raw.get("notes", "")),
                )
            )
    return examples


def extract_claims(text: str) -> set[str]:
    """Pull the checkable claims out of a bullet.

    Every number, plus every token that names something specific:

    - contains a digit, is an acronym, or is inner-capitalised (PyTorch, BiLSTM, 18-class)
    - or is an ordinary capitalised word that is **not** the first word of its sentence

    That last rule is the important one and was missing at first. Restricting detection to
    digits, acronyms and inner capitals looked reasonable but skipped exactly the case that
    matters: "Kubernetes" and "Terraform" are plainly capitalised, so an invented
    technology sailed through while PyTorch would have been caught. Position within the
    sentence is what separates a named tool from the verb a bullet opens with, since the
    rewrite prompt asks for bullets that start with a past-tense verb.
    """
    claims: set[str] = {match.group().lower() for match in _NUMBER_RE.finditer(text)}

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        tokens = list(_TECH_TOKEN_RE.finditer(sentence))
        for index, match in enumerate(tokens):
            token = match.group().strip("./-")
            if len(token) < 2 or token.lower() in _CLAIM_STOP_WORDS:
                continue
            has_digit = any(character.isdigit() for character in token)
            is_acronym = token.isupper()
            inner_capital = any(character.isupper() for character in token[1:])
            # Sentence-initial capitals are skipped: "Designed", "Built", "Achieved".
            named_entity = token[0].isupper() and index > 0
            if has_digit or is_acronym or inner_capital or named_entity:
                claims.add(token.lower())
    return claims


def unsupported_claims(bullet: str, evidence: str) -> set[str]:
    """Claims in the bullet that do not appear in its evidence.

    Substring containment rather than token equality, so "94.28%" matches evidence reading
    "94.28% Binary F1" and "CNN-BiLSTM" matches a hyphenated mention inside a longer line.
    """
    haystack = evidence.lower()
    return {claim for claim in extract_claims(bullet) if claim not in haystack}


def _status_class(status: str) -> str:
    """Collapse the three statuses into have-it or not, which is what a label asserts."""
    if status in (MATCHED, PARTIAL):
        return "present"
    return "absent"


def _requirements_mentioning(keyword: str, requirements: list[Any]) -> list[Any]:
    needle = keyword.lower()
    return [
        requirement
        for requirement in requirements
        if needle in str(getattr(requirement.requirement, "text", "")).lower()
    ]


def evaluate_example(example: TailoringExample, result: Any) -> dict[str, Any]:
    """Score one posting's tailoring result against its labels."""
    matches = list(result.matches)

    label_rows: list[dict[str, Any]] = []
    for keyword, expected in [(k, "present") for k in example.present] + [
        (k, "absent") for k in example.absent
    ]:
        found = _requirements_mentioning(keyword, matches)
        if not found:
            label_rows.append(
                {
                    "keyword": keyword,
                    "expected": expected,
                    "actual": None,
                    "extracted": False,
                    "correct": False,
                }
            )
            continue
        # A keyword can land in more than one requirement; treat it as satisfied if any of
        # them agrees, since the split is the extractor's choice rather than an error.
        actual_classes = {_status_class(match.status) for match in found}
        label_rows.append(
            {
                "keyword": keyword,
                "expected": expected,
                "actual": sorted(actual_classes),
                "extracted": True,
                "correct": expected in actual_classes,
            }
        )

    evidence_by_requirement = {
        match.requirement.id: " ".join(item.chunk.text for item in match.evidence)
        for match in matches
    }
    # The rewriter receives every supported requirement and its evidence in one prompt, so
    # a bullet can legitimately draw on a chunk that belongs to a different requirement.
    # Checking against the union separates the two failure modes: a claim missing from its
    # own requirement's evidence but present elsewhere is *misattributed*, while a claim
    # missing from all of it is *invented*. Only the second is a fabrication.
    all_evidence = " ".join(evidence_by_requirement.values())

    bullet_rows: list[dict[str, Any]] = []
    for bullet in result.bullets:
        evidence = evidence_by_requirement.get(bullet.requirement_id, "")
        unsupported = unsupported_claims(bullet.text, evidence)
        invented = unsupported_claims(bullet.text, all_evidence)
        bullet_rows.append(
            {
                "requirement_id": bullet.requirement_id,
                "text": bullet.text,
                "unsupported_claims": sorted(unsupported),
                "invented_claims": sorted(invented),
                "misattributed": bool(unsupported) and not invented,
                "fabricated": bool(invented),
            }
        )

    return {
        "id": example.id,
        "requirements": len(matches),
        "matched": sum(1 for match in matches if match.status == MATCHED),
        "partial": sum(1 for match in matches if match.status == PARTIAL),
        "missing": sum(1 for match in matches if match.status == MISSING),
        "coverage": result.coverage,
        "recommended_resume": result.recommended_resume,
        "labels": label_rows,
        "bullets": bullet_rows,
        "notes": example.notes,
    }


def run_tailoring_evaluation(
    tailor: Any,
    examples: list[TailoringExample],
) -> list[dict[str, Any]]:
    return [
        evaluate_example(example, tailor.tailor(example.job_description))
        for example in examples
    ]


def summarize_tailoring(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "examples": 0.0,
            "fabrication_rate": 0.0,
            "misattribution_rate": 0.0,
            "clean_bullets": 0.0,
            "bullets": 0.0,
            "classification_accuracy": 0.0,
            "present_accuracy": 0.0,
            "absent_accuracy": 0.0,
            "extraction_recall": 0.0,
            "labels": 0.0,
        }

    labels = [row for record in records for row in record["labels"]]
    bullets = [row for record in records for row in record["bullets"]]
    present = [row for row in labels if row["expected"] == "present"]
    absent = [row for row in labels if row["expected"] == "absent"]

    def share(rows: list[dict[str, Any]], key: str) -> float:
        return sum(1 for row in rows if row[key]) / len(rows) if rows else 0.0

    return {
        "examples": float(len(records)),
        "bullets": float(len(bullets)),
        # Invented: the claim appears in none of the retrieved evidence. This is the
        # number that speaks to the no-fabrication guarantee.
        "fabrication_rate": share(bullets, "fabricated"),
        # Misattributed: grounded in some other requirement's evidence, because the
        # rewriter sees them all in one prompt. Worth reporting, not the same failure.
        "misattribution_rate": share(bullets, "misattributed"),
        "clean_bullets": float(sum(1 for row in bullets if not row["fabricated"])),
        "labels": float(len(labels)),
        "classification_accuracy": share(labels, "correct"),
        # Split because the two failure directions cost different things: a false
        # "present" puts an unsupported claim on a resume, a false "absent" only loses a
        # bullet the candidate had earned.
        "present_accuracy": share(present, "correct"),
        "absent_accuracy": share(absent, "correct"),
        "extraction_recall": share(labels, "extracted"),
    }
