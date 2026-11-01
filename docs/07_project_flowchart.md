> **Note:** this document describes an earlier version of the project. The application now works on simulated UTAS-style transcripts (semesters, courses, marks, attendance, majors) and a semester planner; see the main [README](../README.md) for the current data, results and deployment steps. Figures and charts below are out of date.

## Project Flowchart

Below is a simple flowchart describing the main data and control flow of the Academic Performance Monitoring & Prediction System.

```mermaid
flowchart TD
  A[CSV student data<br/>(`app/data/*.csv`)] --> B[`app/data/data_loader.py`\n(load, compute derived cols)]
  B --> C[`app/ml/preprocessing.py`\n(feature encode, scale, split)]
  C --> D[Trainer: `app/ml/trainer.py`]
  D -->|train| E1[Linear Regression]
  D -->|train| E2[Logistic Regression]
  D -->|train| E3[Random Forest]
  D -->|train| E4[Decision Tree]
  E1 & E2 & E3 & E4 --> F[`app/ml/predictor.py`\n(predict, metrics)]
  F --> G[`app/routes/predictions.py`\n(render predictions page)]
  G --> H[Templates: `app/templates/predictions.html`]
  H -->|user input| I[User adjusts sliders / selects student]
  I --> G
  F --> J[Model evaluation + plots]\n+  J --> K[Saved images: `docs/images/`]
  style A fill:#f8f9fa,stroke:#333,stroke-width:1px
  style D fill:#eef6ff,stroke:#1e6fff
  style F fill:#fff7e6,stroke:#ff9900
```

Legend:
- Data sources: `app/data/*.csv` → loaded and enriched by `app/data/data_loader.py`.
- Preprocessing: `app/ml/preprocessing.py` prepares features and train/test split.
- Training: `app/ml/trainer.py` trains regression and classification models.
- Prediction & metrics: `app/ml/predictor.py` encapsulates model inference and packing of metrics/results.
- Web layer: `app/routes/predictions.py` serves the `predictions.html` template which shows outputs and accepts user input.
- Plots and evaluation artifacts are saved in `docs/images/` and referenced from documentation pages.

How to view:
- Many Markdown viewers (VS Code with Mermaid extension, GitHub preview via Mermaid rendering plugins, or mermaid.live) can render the diagram.

If you'd like, I can also export this diagram to an SVG/PNG and add it to `docs/images/`.
