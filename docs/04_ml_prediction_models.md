# 04 — ML Prediction Models

## Overview

APMS implements four machine learning models from Scikit-learn. All models are trained **in-memory** when the predictions page is first loaded and cached for the session. They are re-trained automatically whenever student data changes (add/edit/delete).

---

## Feature Engineering

### Input Features (10 features total)

| Feature                  | Type        | Notes                                           |
|--------------------------|-------------|------------------------------------------------|
| `attendance_pct`         | Continuous  | Original column                                |
| `math`                   | Continuous  | Subject score 0–100                            |
| `science`                | Continuous  | Subject score 0–100                            |
| `english`                | Continuous  | Subject score 0–100                            |
| `history`                | Continuous  | Subject score 0–100                            |
| `cs`                     | Continuous  | Subject score 0–100                            |
| `assignments_submitted`  | Integer     | Count 0–12                                     |
| `study_hours_per_week`   | Continuous  | Hours per week                                 |
| `previous_gpa`           | Continuous  | 0.0–4.0                                        |
| `gender_encoded`         | Binary      | Male = 1, Female = 0 (Label Encoding)          |

### Preprocessing Steps

1. **Label Encoding** — `gender` column converted to 0/1
2. **Missing value imputation** — median imputation per column
3. **Standard Scaling** — `StandardScaler` (zero mean, unit variance) applied to all features
4. **Train/Test Split** — 80% training, 20% testing, stratified by `pass_fail`, `random_state=42`

---

## Models

### 1. Linear Regression

**Task:** Regression — predicts the student's `avg_score` (continuous value)

**Algorithm:** Ordinary Least Squares fitted to the training set

**Why this model:** Interpretable baseline that quantifies the linear relationship between features and score.

**Output:** Predicted score (0–100) → converted to grade letter (A/B/C/D/F)

**Evaluation Metrics:**
| Metric | Description                                           |
|--------|-------------------------------------------------------|
| RMSE   | Root Mean Squared Error — lower is better             |
| R²     | Coefficient of determination — closer to 1 is better  |

---

### 2. Logistic Regression

**Task:** Classification — predicts `pass_fail` (binary: 0 = Fail, 1 = Pass)

**Algorithm:** Logistic regression with L2 regularization (`max_iter=1000`)

**Why this model:** Probabilistic linear classifier, fast and interpretable.

**Output:** Pass / Fail + confidence probability

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

### 3. Random Forest

**Task:** Classification — predicts `pass_fail`

**Algorithm:** Ensemble of 100 decision trees with bootstrap sampling and random feature selection

**Hyperparameters:** `n_estimators=100`, `random_state=42`

**Why this model:** Robust to overfitting, handles non-linear relationships, typically achieves highest accuracy.

**Output:** Pass / Fail + confidence probability

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

### 4. Decision Tree

**Task:** Classification — predicts `pass_fail`

**Algorithm:** CART decision tree with Gini impurity

**Hyperparameters:** `max_depth=6` (limits overfitting), `random_state=42`

**Why this model:** Fully interpretable — can be visualised as a set of if/then rules.

**Output:** Pass / Fail + confidence probability

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## Metric Definitions

| Metric    | Formula                                         | Interpretation                      |
|-----------|-------------------------------------------------|-------------------------------------|
| Accuracy  | (TP + TN) / Total                               | Overall correct predictions         |
| Precision | TP / (TP + FP)                                  | Of predicted passes, how many passed |
| Recall    | TP / (TP + FN)                                  | Of actual passes, how many predicted |
| F1        | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision & recall |
| RMSE      | √(mean((ŷ - y)²))                               | Average prediction error in score units |
| R²        | 1 - SS_res/SS_tot                               | Variance explained by the model      |

Where: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative

---

## ML Pipeline Code Flow

```
load_students()          → pandas DataFrame (100+ rows)
       │
preprocess()             → StandardScaler, LabelEncoder, train_test_split
       │
train_all_models()       → 4 trained model objects + metrics dict
       │
_model_cache             → in-memory cache (avoids re-training every request)
       │
predict_student(record)  → preprocess_single() → model.predict() → results dict
```

---

## Using the Predictions Page

1. Navigate to **Predictions** (`/predictions`)
2. The right panel shows **Model Evaluation Metrics** — trained on the full dataset
3. Adjust the sliders on the left panel (attendance, scores, study hours, etc.)
4. Click **Run Predictions** — all four models run simultaneously
5. Results appear as cards showing predicted grade (Linear Regression) and pass/fail with confidence (classification models)
6. A grouped bar chart compares accuracy, precision, recall, and F1 across the three classification models

---

*Previous: [03 — Performance Analysis](03_performance_analysis.md) | Next: [05 — Deployment Guide](05_deployment_guide.md)*
