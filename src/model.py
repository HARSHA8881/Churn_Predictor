import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

LR_PATH  = os.path.join(MODELS_DIR, "logistic_regression.pkl")
DT_PATH  = os.path.join(MODELS_DIR, "decision_tree.pkl")
RF_PATH  = os.path.join(MODELS_DIR, "random_forest.pkl")
GB_PATH  = os.path.join(MODELS_DIR, "gradient_boosting.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "minmax_scaler.pkl")


def train_models(X, Y):
    """
    Phase 5: Split data and train 4 classifiers with tuned hyperparameters.

    Accuracy improvements applied:
    - class_weight='balanced'  → corrects the ~20% churn class imbalance
    - Tuned max_depth / n_estimators / learning_rate per model
    - Random Forest + Gradient Boosting added for ensemble power
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, train_size=0.7, random_state=42, stratify=Y
    )

    models = {
        "lr": LogisticRegression(
            C=10.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        ),
        "dt": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=15,
            min_samples_leaf=4,
            random_state=42,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        ),
    }

    for m in models.values():
        m.fit(x_train, y_train)

    return models["lr"], models["dt"], models["rf"], models["gb"], x_train, x_test, y_train, y_test


def save_models(log, dt, rf, gb, minmax):
    """Phase 6: Persist all trained artifacts."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(log,    LR_PATH)
    joblib.dump(dt,     DT_PATH)
    joblib.dump(rf,     RF_PATH)
    joblib.dump(gb,     GB_PATH)
    joblib.dump(minmax, SCALER_PATH)


def load_saved_models():
    """Load all saved model artifacts. Returns None tuple if not found."""
    try:
        log    = joblib.load(LR_PATH)
        dt     = joblib.load(DT_PATH)
        rf     = joblib.load(RF_PATH)
        gb     = joblib.load(GB_PATH)
        minmax = joblib.load(SCALER_PATH)
        return log, dt, rf, gb, minmax
    except FileNotFoundError:
        return None, None, None, None, None


def evaluate_model(model, x_train, y_train, x_test, y_test, threshold=0.5):
    """Compute all evaluation metrics for a trained model."""
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="f1")
    y_probs   = model.predict_proba(x_test)[:, 1]
    y_pred    = (y_probs >= threshold).astype(int)

    return {
        "Accuracy":        accuracy_score(y_test, y_pred),
        "CV Mean F1":      cv_scores.mean(),
        "Precision":       precision_score(y_test, y_pred, zero_division=0),
        "Recall":          recall_score(y_test, y_pred, zero_division=0),
        "F1 Score":        f1_score(y_test, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "FPR":             roc_curve(y_test, y_probs)[0],
        "TPR":             roc_curve(y_test, y_probs)[1],
        "AUC":             auc(*roc_curve(y_test, y_probs)[:2]),
    }
