"""
Hand-coded Bayes' Theorem on the real student records:

    P(Fail | Evidence) = P(Evidence | Fail) x P(Fail) / P(Evidence)

The evidence values are read straight from the data, so every number on the page can be
traced back to simple counts. Only records with a known final mark are used.
"""

import pandas as pd

from app import grading


def _term1_band(row) -> str:
    mark = float(row["t1_mark"])
    return ("Term 1 under 40" if mark < 40 else "Term 1 40-49" if mark < 50 else "Term 1 50-59" if mark < 60
            else "Term 1 60-69" if mark < 70 else "Term 1 70+")


def _trend_band(row) -> str:
    delta = float(row["t2_mark"]) - float(row["t1_mark"])
    return "Marks dropping (5+)" if delta <= -5 else "Marks slipping (2-5)" if delta < -2 else \
        "Marks improving (2+)" if delta >= 2 else "Marks stable"


# (title, function turning a record into an evidence label, label order)
EVIDENCE_TYPES = [
    ("Term 1 mark", _term1_band, ["Term 1 under 40", "Term 1 40-49", "Term 1 50-59", "Term 1 60-69", "Term 1 70+"]),
    ("Change from term 1 to term 2", _trend_band,
     ["Marks dropping (5+)", "Marks slipping (2-5)", "Marks stable", "Marks improving (2+)"]),
    ("Absences", lambda r: grading.absence_band(r["absences"]), [b[2] for b in grading.ABSENCE_BANDS]),
    ("Course", lambda r: str(r["course"]), ["Mathematics", "Portuguese"]),
]


def bayes_table(df: pd.DataFrame) -> dict:
    if df.empty or "fail" not in df.columns:
        return {"prior": None, "by_evidence": [], "worked_examples": []}
    df = df[df["fail"].notna()]
    total = len(df)
    positives = int((df["fail"] == 1).sum())
    prior = positives / total if total else 0.0

    rows = []
    for title, labeller, order in EVIDENCE_TYPES:
        labels = df.apply(labeller, axis=1)
        for evidence in order:
            mask = labels == evidence
            n_evidence = int(mask.sum())
            n_both = int(((df["fail"] == 1) & mask).sum())
            p_e_given_d = n_both / positives if positives else 0.0
            p_e = n_evidence / total if total else 0.0
            posterior = (p_e_given_d * prior) / p_e if p_e else 0.0
            rows.append({
                "evidence_type": title, "evidence": evidence, "students": n_evidence,
                "p_e_given_dropout": round(p_e_given_d, 4), "p_evidence": round(p_e, 4),
                "p_dropout_given_e": round(posterior, 4),
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
                f"P(Fail | {best['evidence']}) = "
                f"({best['p_e_given_dropout']} x {prior}) / {best['p_evidence']} = {best['p_dropout_given_e']}",
            ],
        })
    return examples
