"""Train a reproducible employee mental-health demo classifier.

The preprocessing is fitted only on the training partition and serialized
with the estimator as one scikit-learn Pipeline.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Remote.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "mental_health_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

TARGET = "Mental_Health_Condition"
NUMERIC_FEATURES = ["Hours_Worked_Per_Week"]
ORDINAL_CATEGORIES = [
    ["22-32", "32-42", ">42"],
    ["No", "Yes"],
    ["Poor", "Average", "Good"],
    ["Low", "Medium", "High"],
    ["Not At All", "Weekly", "Daily"],
]
ORDINAL_FEATURES = [
    "Age",
    "access_to_Mental_Health_Resource",
    "sleep_Quality",
    "stress_Level",
    "physical_Activity",
]
NOMINAL_FEATURES = ["gender", "industry", "Location", "region"]
FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES


def build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ORDINAL_FEATURES,
            ),
            ("nominal", OneHotEncoder(handle_unknown="ignore"), NOMINAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(solver="lbfgs", max_iter=2000)),
        ]
    )


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data[TARGET].isna().any():
        raise ValueError("Target contains missing values.")
    if data[FEATURES].isna().any().any():
        raise ValueError("Features contain missing values; resolve them explicitly.")

    X = data[FEATURES].copy()
    y = data[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target": TARGET,
        "target_classes": sorted(y.unique().tolist()),
        "split": {"test_size": 0.20, "random_state": 42, "stratify": True},
        "model": "LogisticRegression(solver='lbfgs', max_iter=2000)",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved pipeline: {MODEL_PATH}")
    print(f"Saved evaluation: {METRICS_PATH}")


if __name__ == "__main__":
    main()
