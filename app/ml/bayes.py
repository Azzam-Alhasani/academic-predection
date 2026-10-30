"""
Hand-coded Bayes' Theorem on the real student records:

    P(Pass | Evidence) = P(Evidence | Pass) x P(Pass) / P(Evidence)

The evidence values are read straight from the data, so every number on the
page can be traced back to simple counts.
"""

import pandas as pd

PASS_MARK = 10


def _absence_band(absences) -> str:
    absences = float(absences)
    if absences == 0:
        return "None"
    if absences <= 5:
        return "Some (1-5)"
    return "Many (6+)"


# (title, column needed, function turning a row value into an evidence label, label order)
EVIDENCE_TYPES = [
    ("Second-period grade (G2)", "G2",
     lambda v: "G2 passing (10+)" if float(v) >= PASS_MARK else "G2 failing (<10)",
     ["G2 passing (10+)", "G2 failing (<10)"]),
    ("School absences", "absences", _absence_band, ["None", "Some (1-5)", "Many (6+)"]),
    ("Past class failures", "failures",
     lambda v: "No past failures" if float(v) == 0 else "1+ past failures",
     ["No past failures", "1+ past failures"]),
]


def bayes_table(df: pd.DataFrame) -> dict:
    if df.empty or "pass_fail" not in df.columns:
        return {"prior_pass": None, "by_evidence": [], "worked_examples": []}

    total = len(df)
    pass_count = int((df["pass_fail"] == 1).sum())
    prior_pass = pass_count / total if total else 0.0

    rows = []
    for title, column, labeller, order in EVIDENCE_TYPES:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").fillna(0)
        labels = values.apply(labeller)
        for evidence in order:
            mask = labels == evidence
            n_evidence = int(mask.sum())
            n_both = int(((df["pass_fail"] == 1) & mask).sum())
            p_e_given_pass = n_both / pass_count if pass_count else 0.0
            p_e = n_evidence / total if total else 0.0
            posterior = (p_e_given_pass * prior_pass) / p_e if p_e else 0.0
            rows.append({
                "evidence_type": title,
                "evidence": evidence,
                "p_evidence_given_pass": round(p_e_given_pass, 4),
                "p_evidence": round(p_e, 4),
                "p_pass_given_evidence": round(posterior, 4),
            })

    return {
        "prior_pass": round(prior_pass, 4),
        "by_evidence": rows,
        "worked_examples": _worked_examples(rows, prior_pass),
    }


def _worked_examples(rows, prior_pass) -> list[dict]:
    grouped = {}
    for r in rows:
        grouped.setdefault(r["evidence_type"], []).append(r)

    examples = []
    for evidence_type, entries in grouped.items():
        best = max(entries, key=lambda e: e["p_pass_given_evidence"])
        prior = round(prior_pass, 4)
        examples.append({
            "evidence_type": evidence_type,
            "evidence": best["evidence"],
            "prior_pass": prior,
            "p_evidence_given_pass": best["p_evidence_given_pass"],
            "p_evidence": best["p_evidence"],
            "p_pass_given_evidence": best["p_pass_given_evidence"],
            "formula_lines": [
                f"P(Pass) = {prior}",
                f"P({best['evidence']} | Pass) = {best['p_evidence_given_pass']}",
                f"P({best['evidence']}) = {best['p_evidence']}",
                f"P(Pass | {best['evidence']}) = "
                f"({best['p_evidence_given_pass']} x {prior}) / "
                f"{best['p_evidence']} = {best['p_pass_given_evidence']}",
            ],
        })
    return examples
