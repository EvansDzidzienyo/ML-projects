# Employee Mental Health Condition Prediction

An educational machine-learning project exploring associations between employee and workplace characteristics and a dataset outcome label.

![Project visual](FRM.jpg)

## Project scope

The repository contains a cleaned dataset, model-training notebook, saved Logistic Regression model, one-hot encoder and Streamlit prototypes. The training work compares Logistic Regression, Random Forest, Gradient Boosting, Support Vector Classifier, Decision Tree and K-Nearest Neighbours.

The original project notes report an F1 score of 86.4% for Logistic Regression. This figure is retained as a historical reported result; it has **not been independently reproduced as part of this documentation update**. Before using it as a portfolio performance claim, rerun the notebook from a clean environment and document the test split, class-wise precision/recall, confusion matrix and preprocessing procedure.

## Repository files

| File | Purpose |
|---|---|
| `Remote.csv` | Project dataset |
| `Remote new-Copy1.ipynb` | Exploratory/model-training notebook |
| `model.pkl` | Serialized trained estimator |
| `onehot_encoder.pkl` | Categorical feature encoder |
| `Mental_Health.py` | Primary Streamlit interface |
| `new.py`, `new_1.py` | Earlier interface prototypes |

## Run locally

Use Python in a virtual environment. Install compatible versions of Streamlit, pandas, numpy, scikit-learn and joblib that match the environment used to serialize the model.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install streamlit pandas numpy scikit-learn joblib
streamlit run Mental_Health.py
```

The serialized model and encoder must be compatible with the installed scikit-learn version. Model files created with a different version may fail to load or behave differently.

## Important limitations

- This is an experimental educational project, **not a clinical tool, mental-health screening instrument or diagnostic system**.
- Predictions should not be used for hiring, employee evaluation, treatment decisions or other consequential decisions.
- The F1 score in the original project notes has not been independently verified in this branch.
- The saved estimator and encoder should be packaged with a single reproducible preprocessing pipeline. The alternate scripts are prototypes and may not share identical input schemas.
- Dataset provenance, consent, representativeness, target definition, class balance and potential demographic/workplace bias should be documented before drawing substantive conclusions.

## Next technical improvements

1. Rebuild training and inference as one scikit-learn Pipeline.
2. Add a pinned, tested `requirements.txt` and reproducible train/test evaluation.
3. Add automated checks that the app feature names and order match the fitted pipeline.
4. Document dataset source, target encoding, limitations and model validation.
