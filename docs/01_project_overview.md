# 01 — Project Overview

## Academic Performance Monitoring & Prediction System (APMS)

### Vision

APMS is a web-based application that gives educators and academic administrators a single unified interface to:

1. **Manage** student academic records (add, edit, view, delete)
2. **Monitor** class-wide and individual-level performance trends through interactive charts
3. **Predict** future student outcomes using four machine learning models
4. **Identify** at-risk students automatically based on attendance and score thresholds

---

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        Browser (UI)                        │
│       Bootstrap 5 Dark  ·  Chart.js  ·  FontAwesome        │
└────────────────────────┬───────────────────────────────────┘
                         │ HTTP (Flask)
┌────────────────────────▼───────────────────────────────────┐
│                    Flask Application                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ /        │  │/students │  │/analytics│  │/prediction│  │
│  │ Dashboard│  │  CRUD    │  │  Charts  │  │  ML Models│  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Data Layer (Pandas + CSV)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              ML Layer (Scikit-learn)                 │  │
│  │  Linear Reg · Logistic Reg · Random Forest · DecTree│  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   students.csv      │
              │ (persistent storage)│
              └─────────────────────┘
```

---

### Tech Stack

| Layer       | Technology         | Version  | Purpose                              |
|-------------|-------------------|----------|--------------------------------------|
| Web Server  | Flask             | 3.0.x    | HTTP routing, templating             |
| Data        | Pandas            | 2.2.x    | CSV I/O, aggregations                |
| ML          | Scikit-learn      | 1.5.x    | Model training & evaluation          |
| Numerics    | NumPy             | 1.26.x   | Array operations                     |
| Frontend    | Bootstrap 5       | 5.3.x    | Responsive dark dashboard UI         |
| Charts      | Chart.js          | 4.4.x    | Interactive visualizations           |
| Icons       | Font Awesome      | 6.5.x    | UI icons                             |
| Config      | python-dotenv     | 1.0.x    | Environment variable loading         |
| Production  | Gunicorn          | 22.x     | WSGI server for deployment           |

---

### Feature Summary

| Feature                        | Location              |
|--------------------------------|-----------------------|
| KPI cards (students, GPA, etc) | `/` Dashboard         |
| Grade distribution chart       | `/` Dashboard         |
| Subject radar chart            | `/` Dashboard         |
| Study hours trend chart        | `/` Dashboard         |
| Student list with search       | `/students`           |
| Add / Edit / Delete student    | `/students`           |
| Per-student score radar        | `/students/<id>`      |
| At-risk badge                  | `/students/<id>`      |
| Top/bottom performer tables    | `/analytics`          |
| GPA distribution histogram     | `/analytics`          |
| Gender subject comparison      | `/analytics`          |
| Attendance vs score scatter    | `/analytics`          |
| Pass/fail prediction           | `/predictions`        |
| Score regression prediction    | `/predictions`        |
| Model evaluation metrics       | `/predictions`        |
| Model comparison chart         | `/predictions`        |

---

### Project Structure

```
academic-performance-predictor/
├── app/
│   ├── __init__.py          ← Flask app factory
│   ├── routes/
│   │   ├── main.py          ← Dashboard
│   │   ├── students.py      ← Student CRUD
│   │   ├── analytics.py     ← Analytics
│   │   └── predictions.py   ← ML predictions
│   ├── ml/
│   │   ├── preprocessing.py ← Feature engineering
│   │   ├── trainer.py       ← Model training
│   │   └── predictor.py     ← Prediction interface
│   ├── data/
│   │   ├── data_loader.py           ← Data access layer
│   │   ├── generate_sample_data.py  ← Sample data generator
│   │   └── sample_students.csv      ← 100 demo students
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── students.html
│   │   ├── student_detail.html
│   │   ├── analytics.html
│   │   └── predictions.html
│   └── static/
│       ├── css/style.css
│       └── js/main.js
├── docs/                    ← Project documentation
├── config.py                ← App configuration
├── run.py                   ← Entry point
├── requirements.txt         ← All Python dependencies
├── .gitignore
└── README.md
```

---

*Next: [02 — Data Collection & Management](02_data_collection_management.md)*
