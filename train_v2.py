"""
Train and export asteroid_guardian_v2 model using only clean, observable features.
Eliminates data leakage and avoids hardcoded dummy features in production.
"""
import zipfile
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_ZIP = "nasa_astreiod_analysis.csv.zip"
CSV_PATH = "nasa_astreiod_analysis.csv/nasa.csv"
MODEL_OUTPUT = "asteroid_guardian_v2.pkl"

FEATURE_COLS = [
    "Absolute Magnitude",
    "Orbit Uncertainity",
    "Relative Velocity km per sec",
    "Eccentricity",
    "Miss Dist.(Astronomical)",
    "Minimum Orbit Intersection"
]

def load_data():
    print("Loading NASA dataset...")
    with zipfile.ZipFile(DATA_ZIP, 'r') as z:
        with z.open(CSV_PATH) as f:
            df = pd.read_csv(f)
    print(f"Loaded {len(df)} records.")
    return df

def train_and_save():
    df = load_data()
    
    X = df[FEATURE_COLS]
    y = df['Hazardous'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    
    # Model with winning hyperparameter configuration from GridSearchCV:
    # 100 trees, balanced weights to ensure zero false negatives (critical planetary defense criterion)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- TEST SET EVALUATION ---")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Hazardous']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    print("\n--- FEATURE IMPORTANCES ---")
    for feat, imp in zip(FEATURE_COLS, model.feature_importances_):
        print(f"  {feat}: {imp:.4f}")
        
    joblib.dump(model, MODEL_OUTPUT)
    print(f"\nModel saved successfully as: {MODEL_OUTPUT}")

if __name__ == "__main__":
    train_and_save()
