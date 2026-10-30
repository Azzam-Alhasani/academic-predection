import pandas as pd


SUBJECT_COLS = ["math", "science", "english", "history", "cs"]
ATTENDANCE_BANDS = [
    ("Low", 0, 60),
    ("Medium", 60, 80),
    ("High", 80, 101),
]


def attendance_band(attendance: float) -> str:
    for label, low, high in ATTENDANCE_BANDS:
        if low <= attendance < high:
            return label
    return "High"


def previous_outcome(prev_term_avg: float) -> str:
    return "Pass" if prev_term_avg >= 50 else "Fail"


def bayes_table(df: pd.DataFrame) -> dict:
    if df.empty or "pass_fail" not in df.columns:
        return {"prior_pass": None, "by_evidence": [], "worked_examples": []}

    total = len(df)
    pass_count = int((df["pass_fail"] == 1).sum())
    prior_pass = pass_count / total if total else 0.0

    evidence_rows = []

    if "previous_term_avg" in df.columns:
        df = df.copy()
        df["_prev_outcome"] = df["previous_term_avg"].apply(previous_outcome)
        for evidence in ["Pass", "Fail"]:
            ev_mask = df["_prev_outcome"] == evidence
            ev_count = int(ev_mask.sum())
            pass_and_evidence = int(((df["pass_fail"] == 1) & ev_mask).sum())
            evidence_given_pass = pass_and_evidence / pass_count if pass_count else 0.0
            evidence_prob = ev_count / total if total else 0.0
            posterior = (evidence_given_pass * prior_pass) / evidence_prob if evidence_prob else 0.0
            evidence_rows.append({
                "evidence_type": "Previous-term outcome",
                "evidence": evidence,
                "p_evidence_given_pass": round(evidence_given_pass, 4),
                "p_evidence": round(evidence_prob, 4),
                "p_pass_given_evidence": round(posterior, 4),
            })

    if "attendance_pct" in df.columns:
        df["_att_band"] = df["attendance_pct"].apply(attendance_band)
        for band in ["Low", "Medium", "High"]:
            ev_mask = df["_att_band"] == band
            ev_count = int(ev_mask.sum())
            pass_and_evidence = int(((df["pass_fail"] == 1) & ev_mask).sum())
            evidence_given_pass = pass_and_evidence / pass_count if pass_count else 0.0
            evidence_prob = ev_count / total if total else 0.0
            posterior = (evidence_given_pass * prior_pass) / evidence_prob if evidence_prob else 0.0
            evidence_rows.append({
                "evidence_type": "Attendance band",
                "evidence": band,
                "p_evidence_given_pass": round(evidence_given_pass, 4),
                "p_evidence": round(evidence_prob, 4),
                "p_pass_given_evidence": round(posterior, 4),
            })

    return {
        "prior_pass": round(prior_pass, 4),
        "by_evidence": evidence_rows,
        "worked_examples": _worked_examples(evidence_rows, prior_pass),
    }


def _worked_examples(rows, prior_pass) -> list[dict]:
    examples = []
    types = {}
    for r in rows:
        types.setdefault(r["evidence_type"], []).append(r)

    for evidence_type, entries in types.items():
        valid = [e for e in entries if e.get("p_pass_given_evidence") is not None]
        if not valid:
            continue
        best = max(valid, key=lambda e: e["p_pass_given_evidence"])
        examples.append({
            "evidence_type": evidence_type,
            "evidence": best["evidence"],
            "prior_pass": round(prior_pass, 4),
            "p_evidence_given_pass": best["p_evidence_given_pass"],
            "p_evidence": best["p_evidence"],
            "p_pass_given_evidence": best["p_pass_given_evidence"],
            "formula_lines": [
                f"P(Pass) = {round(prior_pass,4)}",
                f"P({best['evidence']} | Pass) = {best['p_evidence_given_pass']}",
                f"P({best['evidence']}) = {best['p_evidence']}",
                f"P(Pass | {best['evidence']}) = "
                f"({best['p_evidence_given_pass']} × {round(prior_pass,4)}) / "
                f"{best['p_evidence']} = {best['p_pass_given_evidence']}",
            ],
        })
    return examples