"""Train the machine-learning model (Random Forest) and save it."""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import DATA_FILE, FEATURES, METRICS_FILE, MODEL_FILE


def train(verbose=True):
    if not DATA_FILE.exists():
        from generate_data import generate
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(DATA_FILE, index=False)

    df = pd.read_csv(DATA_FILE)
    X, y = df[FEATURES], df["potable"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                   random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "feature_importance": {f: round(float(v), 4) for f, v in zip(FEATURES, model.feature_importances_)},
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))

    if verbose:
        print(f"Trained on {metrics['rows_train']} rows, tested on {metrics['rows_test']} rows")
        print(f"Accuracy : {metrics['accuracy']:.2%}")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Recall   : {metrics['recall']:.2%}")
        print(f"F1 score : {metrics['f1']:.2%}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:", metrics["confusion_matrix"])
        print("Feature importance:", metrics["feature_importance"])
        print(f"Model saved to {MODEL_FILE}")
    return metrics


if __name__ == "__main__":
    train()
