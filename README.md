# 🚀 NEO Guardian: AI-Powered Planetary Defense

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NEO Guardian** is a production-grade machine learning system designed to analyze and predict the threat levels of **Near-Earth Objects (NEOs)**. By applying supervised machine learning onto NASA orbital surveillance datasets, it classifies **Potentially Hazardous Asteroids (PHAs)** with **100% recall** on threat cases to ensure zero missed collision risks.

---

## 🏛️ System Architecture

NEO Guardian implements a **decoupled microservices architecture** that separates machine learning inference from UI presentation:

```
                  ┌──────────────────────────────────────────────┐
                  │          Streamlit Dashboard (Port 8501)     │
                  │   Tactical Radar • Threat Scanner • Orbit UI  │
                  └──────────────────────┬───────────────────────┘
                                         │ HTTP REST
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            FastAPI Backend (Port 8000)       │
                  │  /predict  •  /health  •  /model-info        │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Balanced Random Forest Classifier (v2)    │
                  │        asteroid_guardian_v2.pkl              │
                  └──────────────────────────────────────────────┘
```

- **Backend API (`api.py`)**: High-performance FastAPI service validating inputs with Pydantic schemas, calculating physical diameters using IAU astrophysics equations, and serving prediction probabilities.
- **Frontend Dashboard (`app.py`)**: Real-time Streamlit tactical command center with orbital eccentricity visualization, comparative scale bars vs Earth landmarks, and session CSV audit export.
- **Resilient Fallback**: The frontend automatically connects to the FastAPI backend, with graceful fallback to local model execution if the backend is unavailable.

---

## 🎯 Model & Machine Learning Performance

Trained using 5-fold Stratified `GridSearchCV` on NASA's observational dataset, using **strictly clean observable parameters** to eliminate feature leakage and ensure reproducibility:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Hazardous Recall** | **100.0%** | Zero false negatives — critical for planetary defense |
| **Hazardous Precision** | **98.0%** | Minimal false alarms (3 out of 787 safe asteroids) |
| **Accuracy** | **99.68%** | Evaluated on 938 unseen test samples |
| **ROC-AUC** | **1.0000** | Exceptional rank ordering across decision thresholds |

### The 6 Observable Features:
1. **Minimum Orbit Intersection (MOID)**: Minimum distance between Earth and asteroid orbits (*48.1% importance*).
2. **Absolute Magnitude ($H$)**: Photometric brightness used to estimate physical mass (*34.2% importance*).
3. **Orbit Uncertainty ($U$)**: Observational data quality metric (*10.1% importance*).
4. **Relative Velocity**: Speed at atmospheric entry (*3.0% importance*).
5. **Orbit Eccentricity ($e$)**: Shape/elongation of elliptical path (*2.7% importance*).
6. **Miss Distance**: Closest approach distance (*1.9% importance*).

### Astronomical Diameter Formula
Physical asteroid diameter is derived using the International Astronomical Union (IAU) equation:
$$D = \frac{1329}{\sqrt{Albedo}} \cdot 10^{-0.2H}$$
*(Assuming an optical albedo of 0.15 for stony Near-Earth asteroids)*

---

## ⚡ Quick Start

### 1. Local Setup
```bash
# Clone the repository
git clone https://github.com/theyasassri/Asteroid_prediction_model.git
cd Asteroid_prediction_model

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

#### Option A: Run Full Stack (API + Dashboard)
Terminal 1 (Backend API):
```bash
python api.py
# API runs on: http://localhost:8000 (Swagger docs: http://localhost:8000/docs)
```

Terminal 2 (Streamlit UI):
```bash
streamlit run app.py
# UI runs on: http://localhost:8501
```

#### Option B: Run Dashboard Only
```bash
streamlit run app.py
# The dashboard automatically falls back to local model inference if API is not running
```

---

## 🐳 Docker Deployment

Run the entire decoupled system with a single command via Docker Compose:

```bash
docker-compose up --build
```
- Dashboard available at: `http://localhost:8501`
- REST API available at: `http://localhost:8000`
- Interactive API Docs at: `http://localhost:8000/docs`

---

## 🧪 Automated Testing

Run the test suite covering model integrity, physics formulas, API status codes, and input validation:

```bash
pytest test_app.py -v
```

---

## 📁 Repository Structure

```
Asteroid_prediction_model/
├── api.py                    # Production FastAPI backend REST service
├── app.py                    # Interactive Streamlit planetary defense dashboard
├── asteroid_guardian_v2.pkl  # Trained Random Forest model (clean 6-feature)
├── train_v2.py               # Retraining and export pipeline script
├── test_app.py               # Pytest suite (Model, Physics, API, Pydantic)
├── Dockerfile                # Production container specification
├── docker-compose.yml        # Multi-service orchestration config
├── requirements.txt          # Pinned dependency definitions
├── IMPLEMENTATION_PLAN.md    # Architecture and progress roadmap
└── Astroid.ipynb             # Exploratory Data Analysis & baseline research
```

---

## 👨‍💻 Author & License
- **Author**: Yasassri Ekanayake
- **License**: MIT
