from pathlib import Path
import sys

import joblib
import pandas as pd

# Ensure the repository root is importable when pytest runs from the tests directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import DATA_PATH, FEATURES, TARGET, build_pipeline


def test_pipeline_trains_and_predicts_on_dataset_rows(tmp_path: Path):
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data[TARGET].astype(str)

    pipeline = build_pipeline()
    pipeline.fit(X.iloc[:1000], y.iloc[:1000])

    predictions = pipeline.predict(X.iloc[1000:1010])
    assert len(predictions) == 10
    assert set(predictions).issubset(set(y.unique()))

    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, artifact)
    loaded = joblib.load(artifact)
    reloaded_predictions = loaded.predict(X.iloc[1000:1010])
    assert list(reloaded_predictions) == list(predictions)


def test_pipeline_accepts_one_unseen_nominal_category():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data[TARGET].astype(str)
    pipeline = build_pipeline().fit(X.iloc[:1000], y.iloc[:1000])

    sample = X.iloc[[1000]].copy()
    sample["industry"] = "Previously unseen industry"
    prediction = pipeline.predict(sample)
    assert len(prediction) == 1
