"""
Regenerate docs/model_results.json from the real dataset.

Run from the project root:  python -m app.ml.export_results
Metrics are 5-fold grouped cross-validation scores (see app/ml/trainer.py).
"""

import json
import os

from app import create_app
from app.data.data_loader import load_students
from app.ml.preprocessing import prepare
from app.ml.trainer import train_all_models

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "model_results.json")


def main():
    app = create_app("development")
    with app.app_context():
        df = load_students()
        X, y_reg, y_clf, groups = prepare(df)
        results = train_all_models(X, y_reg, y_clf, groups)
    out = {"n_students": int(len(X)), "pass_rate": round(float(y_clf.mean()), 3)}
    for key, r in results.items():
        out[key] = {k: v for k, v in r.items() if k not in ("model", "description", "type", "name")}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
