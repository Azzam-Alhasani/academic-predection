"""
Hand-coded Bayes' Theorem on the real student records:

    P(Dropout | Evidence) = P(Evidence | Dropout) x P(Dropout) / P(Evidence)

The evidence values are read straight from the data, so every number on the
page can be traced back to simple counts.
"""

import pandas as pd


def _s1_completion_band(row) -> str:
    enrolled = float(row["s1_enrolled"])
    if enrolled <= 0:
        return "No courses in semester 1"
    rate = float(row["s1_approved"]) / enrolled
    if rate >= 0.75:
        return "Passed 75%+ of semester-1 courses"
    if rate >= 0.5:
        return "Passed 50-74% of semester-1 courses"
    return "Passed under 50% of semester-1 courses"


# (title, function turning a record into an evidence label, label order)
EVIDENCE_TYPES = [
    ("Semester-1 course completion", _s1_completion_band,
     ["Passed 75%+ of semester-1 courses", "Passed 50-74% of semester-1 courses",
      "Passed under 50% of semester-1 courses", "No courses in semester 1"]),
    ("Tuition fees", lambda r: "Fees up to date" if r["tuition_paid"] == "yes" else "Fees overdue",
     ["Fees up to date", "Fees overdue"]),
    ("Scholarship", lambda r: "Scholarship holder" if r["scholarship"] == "yes" else "No scholarship",
     ["Scholarship holder", "No scholarship"]),
    ("Debt", lambda r: "Has debt" if r["debtor"] == "yes" else "No debt", ["Has debt", "No debt"]),
]


def bayes_table(df: pd.DataFrame) -> dict:
    if df.empty or "dropout" not in df.columns:
        return {"prior": None, "by_evidence": [], "worked_examples": []}

    total = len(df)
    positives = int((df["dropout"] == 1).sum())
    prior = positives / total if total else 0.0

    rows = []
    for title, labeller, order in EVIDENCE_TYPES:
        labels = df.apply(labeller, axis=1)
        for evidence in order:
            mask = labels == evidence
            n_evidence = int(mask.sum())
            n_both = int(((df["dropout"] == 1) & mask).sum())
            p_e_given_d = n_both / positives if positives else 0.0
            p_e = n_evidence / total if total else 0.0
            posterior = (p_e_given_d * prior) / p_e if p_e else 0.0
            rows.append({
                "evidence_type": title,
                "evidence": evidence,
                "students": n_evidence,
                "p_e_given_dropout": round(p_e_given_d, 4),
                "p_evidence": round(p_e, 4),
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
            "evidence_type": evidence_type,
            "evidence": best["evidence"],
            "posterior": best["p_dropout_given_e"],
            "formula_lines": [
                f"P(Dropout) = {prior}",
                f"P({best['evidence']} | Dropout) = {best['p_e_given_dropout']}",
                f"P({best['evidence']}) = {best['p_evidence']}",
                f"P(Dropout | {best['evidence']}) = "
                f"({best['p_e_given_dropout']} x {prior}) / {best['p_evidence']} = {best['p_dropout_given_e']}",
            ],
        })
    return examples
