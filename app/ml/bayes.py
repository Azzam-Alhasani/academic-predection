"""
Hand-coded Bayes' Theorem on the student records:

    P(Fail | Evidence) = P(Evidence | Fail) x P(Fail) / P(Evidence)

Every number comes straight from counts of completed course attempts, so it can be checked by hand.
"""

import pandas as pd

from app import grading


def _cgpa_band(row) -> str:
    c = row["prior_cgpa"]
    if pd.isna(c):
        return "First semester (no CGPA yet)"
    return "CGPA below 2.0" if c < 2.0 else "CGPA 2.0-2.9" if c < 3.0 else "CGPA 3.0+"


def _tier(row, tiers) -> str:
    return f"{tiers.get(row['course_code'], 'Medium')} course"


def evidence_types(tiers: dict):
    return [
        ("Attendance in the course", lambda r: grading.attendance_band(r["attendance"]),
         list(reversed(grading.ATTENDANCE_ORDER))),
        ("Course difficulty (measured from marks)", lambda r: _tier(r, tiers), ["Hard course", "Medium course", "Easy course"]),
        ("CGPA before the semester", _cgpa_band,
         ["CGPA below 2.0", "CGPA 2.0-2.9", "CGPA 3.0+", "First semester (no CGPA yet)"]),
        ("Credit load that semester", lambda r: grading.load_status(r["load_credits"]),
         ["Light Load", "Normal Load", "Overload"]),
    ]


def bayes_table(frame: pd.DataFrame, tiers: dict) -> dict:
    if frame is None or frame.empty:
        return {"prior": None, "by_evidence": [], "worked_examples": []}
    df = frame[frame["attempted"] > 0]
    total = len(df)
    positives = int((df["fail"] == 1).sum())
    prior = positives / total if total else 0.0

    rows = []
    for title, labeller, order in evidence_types(tiers):
        labels = df.apply(labeller, axis=1)
        for evidence in order:
            mask = labels == evidence
            n_evidence = int(mask.sum())
            if n_evidence < 20:
                continue
            n_both = int(((df["fail"] == 1) & mask).sum())
            p_e_given_d = n_both / positives if positives else 0.0
            p_e = n_evidence / total
            rows.append({
                "evidence_type": title, "evidence": evidence, "students": n_evidence,
                "p_e_given_dropout": round(p_e_given_d, 4), "p_evidence": round(p_e, 4),
                "p_dropout_given_e": round((p_e_given_d * prior) / p_e, 4),
            })
    return {"prior": round(prior, 4), "by_evidence": rows, "worked_examples": _worked_examples(rows, prior)}


def _worked_examples(rows, prior) -> list[dict]:
    grouped = {}
    for r in rows:
        grouped.setdefault(r["evidence_type"], []).append(r)
    examples = []
    for evidence_type, entries in grouped.items():
        best = max(entries, key=lambda e: e["p_dropout_given_e"])
        examples.append({
            "evidence_type": evidence_type, "evidence": best["evidence"], "posterior": best["p_dropout_given_e"],
            "formula_lines": [
                f"P(Fail) = {prior}",
                f"P({best['evidence']} | Fail) = {best['p_e_given_dropout']}",
                f"P({best['evidence']}) = {best['p_evidence']}",
                f"P(Fail | {best['evidence']}) = ({best['p_e_given_dropout']} x {prior}) / {best['p_evidence']} = {best['p_dropout_given_e']}",
            ],
        })
    return examples
