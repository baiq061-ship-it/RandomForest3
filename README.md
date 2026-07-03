# Streamlit Deployment Package

This folder is ready to use as the root of a Streamlit Community Cloud repository.

## Files

- `app.py`: Streamlit entrypoint.
- `best_model.joblib`: deployed RandomForest pipeline with MissForest preprocessing.
- `missforest_imputer.py`: custom class required to load the model.
- `calculator_metadata.json`, `model_metadata.json`, `run_config_and_columns.json`: model metadata.
- `feature_schema.json`: input labels, defaults, and batch-column aliases.
- `requirements.txt`: Python dependencies for Streamlit Cloud.
- `sample_input.csv`: example batch input.
- `smoke_test.py`: local model-loading check.

## Local Test

```bash
pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

## Streamlit Cloud

1. Put all files in this folder at the root of a GitHub repository.
2. Create a new Streamlit Community Cloud app.
3. Set the entrypoint to `app.py`.
4. Use the same Python version as the model environment when possible, and let Streamlit install dependencies from `requirements.txt`.

## Batch CSV Columns

Required columns:

```text
Symptom duration, Heart rate, K+, Age, First episode
```

For potassium, `K+` and `K⁺` are both accepted. The app displays potassium as `K⁺`.

Single-patient prediction results include a dynamic SHAP force plot for the positive class probability.

This model is for research and clinical review only. It should not be used as the sole basis for patient-care decisions.
