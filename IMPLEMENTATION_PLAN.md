# NEO Guardian - Implementation & Productionization Plan

This document outlines the step-by-step roadmap to transform the **NEO Guardian** prototype into a production-ready, interview-grade application. It addresses critical architectural flaws, resolves feature mismatches, and establishes professional engineering practices.

## Phase 1: Model & Pipeline Refactoring (Core Engineering)

### 1.1 Fix Feature Leakage & Redefine the Feature Space
Currently, the model is trained on 17 features, many of which (like `Est Dia in KM(max)`) are directly correlated with the target or not easily observable for a new asteroid. 
*   **Action:** Update `Astroid.ipynb` to train **only** on the 6 key observational features that the dashboard collects:
    1.  Absolute Magnitude (H)
    2.  Orbit Uncertainty (U)
    3.  Relative Velocity (km/s)
    4.  Eccentricity (e)
    5.  Miss Distance (AU)
    6.  Minimum Orbit Intersection Distance (MOID)
*   **Action:** Drop all other columns before splitting the data to ensure the model doesn't rely on data unavailable at inference time.

### 1.2 Retrain and Export the New Model
*   **Action:** Rerun the `GridSearchCV` on the reduced 6-feature dataset. 
*   **Action:** Ensure `class_weight='balanced'` is used and optimized for **Recall**.
*   **Action:** Re-export the trained model as `asteroid_guardian_v2.pkl`.

### 1.3 Fix the Dashboard (`app.py`) Inference Logic
*   **Action:** Remove the hardcoded dummy features in `app.py` (Lines 133-137).
*   **Action:** Map the 6 UI sliders directly to the input DataFrame expected by the new `asteroid_guardian_v2.pkl` model.
*   **Action:** Update the feature importance chart (Tab 3) to dynamically pull `model.feature_importances_` rather than using hardcoded values.

---

## Phase 2: Production Readiness & Architecture

### 2.1 Dependency Management
*   **Action:** Create a `requirements.txt` file at the root of the repository to pin exact library versions.
    ```text
    streamlit==1.32.0
    scikit-learn==1.4.1.post1
    pandas==2.2.1
    numpy==1.26.4
    joblib==1.3.2
    ```

### 2.2 API Separation (Backend vs. Frontend)
To demonstrate architectural maturity, the ML inference should be decoupled from the UI.
*   **Action:** Create a new file `api.py` using **FastAPI**.
*   **Action:** Define a `POST /predict` endpoint that takes a Pydantic model (the 6 features) and returns the hazard classification, probability, and estimated diameter.
*   **Action:** Update `app.py` to make a REST call to `http://localhost:8000/predict` instead of loading the `.pkl` file directly.

### 2.3 Containerization
*   **Action:** Create a `Dockerfile` to package both the API and the Streamlit app.
    ```dockerfile
    FROM python:3.9-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    # Use a script or docker-compose to run both FastAPI and Streamlit
    ```

---

## Phase 3: Code Quality & Automated Testing

### 3.1 Unit Testing
*   **Action:** Create a `tests/` directory.
*   **Action:** Write tests using `pytest` to verify:
    *   The model loads correctly.
    *   The IAU diameter calculation formula returns expected mathematical results.
    *   The FastAPI `/predict` endpoint handles valid and invalid payloads correctly.

### 3.2 Code Formatting & Linting
*   **Action:** Format the codebase using `black`.
*   **Action:** Ensure imports are sorted with `isort` and code passes `flake8` checks.

---

## Execution Checklist

- [x] **Step 1:** Modify/script training on exactly 6 clean observable features without data leakage (`train_v2.py`).
- [x] **Step 2:** Export `asteroid_guardian_v2.pkl` (achieved 100% recall on hazardous class with balanced weights).
- [x] **Step 3:** Update `app.py` to use `asteroid_guardian_v2.pkl` with dynamic feature importance and zero dummy variables.
- [x] **Step 4:** Generate `requirements.txt` with pinned dependencies.
- [x] **Step 5:** Create `api.py` (FastAPI backend with `/predict`, `/health`, and `/model-info`).
- [x] **Step 6:** Refactor `app.py` to call `api.py` with seamless local model fallback.
- [x] **Step 7:** Write Unit & Integration Tests (`test_app.py` - 7 passed).
- [x] **Step 8:** Add `Dockerfile` and `docker-compose.yml` for containerized deployment.
- [x] **Step 9:** Update `README.md` with production architecture diagrams, performance benchmarks, and setup instructions.
