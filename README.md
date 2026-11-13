# Employee Mental Health Condition Prediction

An educational machine-learning project using the repository's 5,000-row remote-work dataset to demonstrate binary classification. The model predicts the dataset's recorded `Mental_Health_Condition` label (`fit` or `ill`); these labels are dataset values, not clinical assessments.

![Project visual](FRM.jpg)

## Dataset and target

The source file is `Remote.csv` (5,000 observations). The target is `Mental_Health_Condition`, with 3,804 `ill` and 1,196 `fit` records in the committed dataset. The model inputs are age group, hours worked per week, gender, industry, work location, sleep quality, stress level, access to mental-health resources, physical activity and region.

The original notebook fitted categorical encoders on the full dataset before creating its train/test split. That allows information from the test partition to influence preprocessing. The reproducible workflow in `train.py` corrects this by placing preprocessing and Logistic Regression in one scikit-learn Pipeline and fitting that pipeline only on the training partition. The split is stratified, uses 20% for testing and fixes `random_state=42`.

The repository does not currently document an independently verifiable upstream dataset source, collection protocol or consent process. The dataset's introductory notebook describes a Crescent Corp remote-work scenario; that narrative alone should not be treated as independently verified provenance.

## Reproduce the model

Python 3.12 is used by the automated workflow. From the repository root:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
pytest -q
```

Training creates:

- `artifacts/mental_health_pipeline.joblib`: preprocessing and Logistic Regression estimator saved together.
- `artifacts/metrics.json`: test-set accuracy, macro/weighted F1 and class-wise classification report.

To launch the Streamlit interface after training:

```bash
streamlit run app.py
```

The interface uses the same saved pipeline as training, so input encoding is not separately reimplemented in the app.

## Evaluation

On the deterministic stratified 80/20 split (4,000 training rows; 1,000 test rows), the unweighted Logistic Regression pipeline achieved:

| Metric | Test result |
|---|---:|
| Accuracy | 0.761 |
| Macro F1 | 0.432 |
| Weighted F1 | 0.658 |
| F1 for `ill` | 0.864 |
| F1 for `fit` | 0.000 |

**Important interpretation:** the model predicted every test observation as `ill`. Its 76.1% accuracy equals the share of `ill` records in the test set, and it failed to identify any `fit` cases. The 86.4% `ill`-class F1 is therefore not evidence of balanced predictive performance. This explains why a single F1 figure is inadequate. The old notebook's reported 86.4% is consistent with this class-specific result, but does not establish useful performance.

A GitHub Actions workflow installs pinned dependencies, retrains the model from `Remote.csv`, and tests prediction, serialization/reloading and handling of an unseen nominal category. The generated full classification report is saved to `artifacts/metrics.json`.

## Repository files

| File | Purpose |
|---|---|
| `Remote.csv` | Dataset used for the demonstration |
| `Remote new-Copy1.ipynb` | Original exploratory and modelling notebook |
| `train.py` | Reproducible training and evaluation |
| `app.py` | Streamlit interface using the trained pipeline |
| `tests/test_pipeline.py` | Training, inference and serialization tests |
| `Mental_Health.py`, `new.py`, `new_1.py` | Earlier app prototypes retained for reference |
| `model.pkl`, `onehot_encoder.pkl` | Earlier separate model and encoder artefacts retained for reference |

## Limitations and responsible use

- This is an experimental educational demonstration, **not a clinical tool, mental-health screening instrument or diagnostic system**.
- Predictions must not be used for hiring, employee evaluation, treatment decisions or other consequential decisions.
- The dataset's provenance, collection method, consent, representativeness and suitability for real-world inference are not established by the materials in this repository.
- The model learns associations in this dataset. It does not establish causal relationships or determine an individual's mental-health status.
- Performance on this dataset does not establish generalisation to other organisations, countries, populations or time periods.
- The current baseline is not suitable for practical prediction because it does not identify the `fit` class. Further model development and validation are required before even considering a research use.
