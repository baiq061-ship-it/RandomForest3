from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import shap

import missforest_imputer  # noqa: F401  required for loading best_model.joblib


def main() -> None:
    base = Path(__file__).resolve().parent
    model = joblib.load(base / "best_model.joblib")
    metadata = json.loads((base / "calculator_metadata.json").read_text(encoding="utf-8"))
    features = metadata["selected_features"]
    medians = metadata.get("continuous_medians", {})
    threshold = float(metadata.get("best_threshold", 0.5))
    sample = pd.DataFrame([{feature: float(medians.get(feature, 0.0)) for feature in features}])
    probability = float(model.predict_proba(sample)[0, 1])
    predicted_class = int(probability >= threshold)
    missforest = model.named_steps["missforest"]
    analysis = model.named_steps["analysis_pipeline"]
    transformed = analysis.named_steps["preprocess"].transform(missforest.transform(sample))
    explainer = shap.TreeExplainer(analysis.named_steps["model"])
    shap_values = explainer.shap_values(transformed)
    print("Smoke test passed")
    print(f"Probability: {probability:.6f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Predicted class: {predicted_class}")
    print(f"SHAP values shape: {getattr(shap_values, 'shape', 'list')}")


if __name__ == "__main__":
    main()
