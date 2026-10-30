# 05 — Deployment Guide

## Quick Start — One-Click Launchers

### Windows
Double-click **`run.bat`** in the project root. It will:
1. Check that Python is installed
2. Create a virtual environment (`venv/`) if one doesn't exist
3. Install all dependencies from `requirements.txt`
4. Generate `sample_students.csv` if not present
5. Start the server and open `http://127.0.0.1:5000` in your browser

### macOS / Linux
```bash
bash run.sh
```
Same steps as above using `python3` and `venv/bin/activate`.

---

## Prerequisites

- Python 3.10 or newer
- `pip` (included with Python)
- Git (for cloning the repository)
- Internet access (to install packages from PyPI)

---

## Quick Start (Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/Ramendan/academic-performance-predictor.git
cd academic-performance-predictor
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask 3.0.3 — web framework
- Pandas 2.2.2 — data manipulation
- NumPy 1.26.4 — numerical operations
- Scikit-learn 1.5.0 — machine learning
- Matplotlib & Seaborn — (available for offline chart generation)
- python-dotenv — environment variable loading
- Gunicorn — production WSGI server

### 4. Generate sample data (run once)

```bash
python app/data/generate_sample_data.py
```

This creates `app/data/sample_students.csv` with 100 demo students. **Skip this step** if the file already exists.

### 5. Start the development server

```bash
python run.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## Environment Variables

Create a `.env` file in the project root to override defaults (optional):

```env
FLASK_ENV=development
SECRET_KEY=your-secure-secret-key-here
```

> **Security Note:** Never commit your `.env` file. It is already listed in `.gitignore`.

---

## Project Ports

By default the app runs on `0.0.0.0:5000` (accessible from the local network).

To change the port, edit `run.py`:
```python
app.run(debug=True, host="0.0.0.0", port=8080)  # change port here
```

---

## Production Deployment (Gunicorn)

For a production server (Linux/macOS):

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:5000 "run:app"
```

### Using a Systemd Service (Linux)

Create `/etc/systemd/system/apms.service`:

```ini
[Unit]
Description=APMS Flask Application
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/academic-performance-predictor
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 "run:app"
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable apms
sudo systemctl start apms
```

---

## Updating Student Data in Production

The `app/data/students.csv` file can be updated while the app is running. Changes made through the web UI are written to disk immediately.

To reset all data back to the 100-student sample:
```bash
cp app/data/sample_students.csv app/data/students.csv
```

---

## Dependencies Cross-Platform Compatibility

All packages in `requirements.txt` are widely cross-platform:

| Package      | Windows | macOS | Linux |
|-------------|---------|-------|-------|
| Flask        | ✓       | ✓     | ✓     |
| Pandas       | ✓       | ✓     | ✓     |
| NumPy        | ✓       | ✓     | ✓     |
| Scikit-learn | ✓       | ✓     | ✓     |
| Gunicorn     | ✗*      | ✓     | ✓     |

> *Gunicorn does not support Windows natively. On Windows in production, use [Waitress](https://docs.pylonsproject.org/projects/waitress/) instead: `pip install waitress` then `waitress-serve --port=5000 run:app`

---

## Common Issues

| Problem                             | Solution                                                   |
|-------------------------------------|------------------------------------------------------------|
| `ModuleNotFoundError: flask`        | Activate virtual environment, run `pip install -r requirements.txt` |
| `Address already in use`            | Change port in `run.py`, or kill the process using port 5000 |
| Charts not displaying               | Check browser console — ensure CDN URLs are reachable      |
| `students.csv` not found            | Run `python app/data/generate_sample_data.py`              |
| Models show "Not enough data"       | Need at least 10 student records with valid scores         |
| PowerShell execution policy error   | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`  |

---

*Previous: [04 — ML Prediction Models](04_ml_prediction_models.md)*
