import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def train_models(X, Y):
    """Phase 5: Split data and train Logistic Regression + Decision Tree."""
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, train_size=0.7, random_state=0
    )
    log = LogisticRegression(max_iter=1000, random_state=0)
    dt = DecisionTreeClassifier(random_state=0)
    log.fit(x_train, y_train)
    dt.fit(x_train, y_train)
    return log, dt, x_train, x_test, y_train, y_test


def save_models(log, dt, minmax):
    """Phase 6: Persist trained artifacts with joblib."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(log, os.path.join(MODELS_DIR, "modellog.joblib"))
    joblib.dump(dt, os.path.join(MODELS_DIR, "modeldt.joblib"))
    joblib.dump(minmax, os.path.join(MODELS_DIR, "minmaxscaler.joblib"))


def load_saved_models():
    """Load previously saved model artifacts. Returns None tuple if not found."""
    try:
        log = joblib.load(os.path.join(MODELS_DIR, "modellog.joblib"))
        dt = joblib.load(os.path.join(MODELS_DIR, "modeldt.joblib"))
        minmax = joblib.load(os.path.join(MODELS_DIR, "minmaxscaler.joblib"))
        return log, dt, minmax
    except FileNotFoundError:
        return None, None, None


def evaluate_model(model, x_train, y_train, x_test, y_test, threshold=0.5):
    """Compute all evaluation metrics for a trained model."""
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    y_probs = model.predict_proba(x_test)[:, 1]
    ytestPred = (y_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, ytestPred)
    precision = precision_score(y_test, ytestPred, zero_division=0)
    recall = recall_score(y_test, ytestPred, zero_division=0)
    f1 = f1_score(y_test, ytestPred, zero_division=0)
    cm = confusion_matrix(y_test, ytestPred)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    return {
        'Accuracy': accuracy,
        'CV Mean Accuracy': cv_scores.mean(),
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm,
        'FPR': fpr,
        'TPR': tpr,
        'AUC': roc_auc,
    }
