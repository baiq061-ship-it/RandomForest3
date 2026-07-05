from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import missforest_imputer  # noqa: F401  required for loading best_model.joblib


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_model.joblib"
CALCULATOR_METADATA_PATH = APP_DIR / "calculator_metadata.json"
MODEL_METADATA_PATH = APP_DIR / "model_metadata.json"
FEATURE_SCHEMA_PATH = APP_DIR / "feature_schema.json"
POTASSIUM_DISPLAY = "K\u207a"


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = joblib.load(MODEL_PATH)
    calculator_metadata = json.loads(CALCULATOR_METADATA_PATH.read_text(encoding="utf-8"))
    model_metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    feature_schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
    return model, calculator_metadata, model_metadata, feature_schema


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {POTASSIUM_DISPLAY: "K+", "K\uff0b": "K+", "K": "K+"}
    return df.rename(columns={c: aliases.get(str(c).strip(), str(c).strip()) for c in df.columns})


def display_name(name: str) -> str:
    clean = str(name)
    for prefix in ("num__", "num_", "cat__", "cat_", "remainder__"):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
            break
    return clean.replace("K+", POTASSIUM_DISPLAY)


def display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: display_name(column) for column in df.columns})


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1060px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .model-hero {
            border: 1px solid #d8dee9;
            border-left: 5px solid #2563eb;
            padding: 1.05rem 1.15rem;
            margin-bottom: 1rem;
            background: #ffffff;
        }
        .model-title {
            font-size: 1.75rem;
            font-weight: 760;
            line-height: 1.2;
            color: #0f172a;
            margin: 0 0 .35rem 0;
        }
        .model-subtitle {
            color: #475569;
            font-size: .98rem;
            margin: 0;
        }
        .section-label {
            color: #334155;
            font-weight: 700;
            font-size: .94rem;
            margin: .25rem 0 .6rem 0;
        }
        .result-card {
            border: 1px solid #d8dee9;
            padding: .95rem 1rem;
            background: #ffffff;
            margin: .45rem 0 1rem 0;
        }
        .result-value {
            font-size: 2rem;
            font-weight: 780;
            color: #0f172a;
            line-height: 1.05;
        }
        .result-label {
            color: #475569;
            font-size: .9rem;
            margin-top: .25rem;
        }
        .status-high {
            color: #166534;
            font-weight: 720;
        }
        .status-low {
            color: #92400e;
            font-weight: 720;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d8dee9;
            padding: .7rem .8rem;
            background: #ffffff;
        }
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def coerce_model_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = normalize_columns(df)
    missing = [feature for feature in features if feature not in out.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    out = out[features].copy()
    for feature in features:
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    return out


def predict_dataframe(model: Any, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    prob = model.predict_proba(df)[:, 1]
    result = df.copy()
    result["predicted_probability"] = prob
    result["decision_threshold"] = threshold
    result["predicted_class"] = (prob >= threshold).astype(int)
    result["prediction_label"] = result["predicted_class"].map(
        {1: "Likely conversion success", 0: "Lower predicted conversion success"}
    )
    return result


def feature_label(name: str, schema: dict[str, Any]) -> str:
    item = schema["features"].get(name, {})
    label = item.get("label", display_name(name))
    unit = item.get("unit", "")
    return f"{label} ({unit})" if unit else label


def positive_class_shap_values(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if isinstance(values, list):
        return np.asarray(values[1])[0]
    if arr.ndim == 3:
        return arr[0, :, 1]
    if arr.ndim == 2:
        return arr[0]
    return arr


def positive_expected_value(expected_value: Any) -> float:
    expected = np.asarray(expected_value).reshape(-1)
    if expected.size > 1:
        return float(expected[1])
    return float(expected[0])


def render_shap_force_plot(model: Any, frame: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Individual SHAP Explanation</div>', unsafe_allow_html=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        missforest = model.named_steps["missforest"]
        analysis_pipeline = model.named_steps["analysis_pipeline"]
        preprocess = analysis_pipeline.named_steps["preprocess"]
        estimator = analysis_pipeline.named_steps["model"]

        imputed = missforest.transform(frame)
        transformed = preprocess.transform(imputed)
        transformed_array = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        raw_feature_names = list(preprocess.get_feature_names_out())
        feature_names = [display_name(name) for name in raw_feature_names]
        feature_values = transformed_array[0]

        explainer = shap.TreeExplainer(estimator)
        shap_values = positive_class_shap_values(explainer.shap_values(transformed_array))
        expected_value = positive_expected_value(explainer.expected_value)

        plt.figure(figsize=(12, 3.2))
        shap.force_plot(
            expected_value,
            shap_values,
            feature_values,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            link="identity",
            text_rotation=35,
        )
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as exc:
        st.warning("SHAP explanation is temporarily unavailable. The prediction result above is unaffected.")
        with st.expander("SHAP technical details"):
            st.write(str(exc))
        return

    contributions = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "feature_value": feature_values,
                "shap_value": shap_values,
            }
        )
        .sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    with st.expander("SHAP contribution values"):
        st.dataframe(contributions, use_container_width=True, hide_index=True)


def render_single_prediction(
    model: Any,
    calculator_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    features = list(calculator_metadata["selected_features"])
    threshold = float(calculator_metadata.get("best_threshold", 0.5))

    st.markdown('<div class="section-label">Predictor Entry</div>', unsafe_allow_html=True)
    with st.form("single_prediction_form"):
        values: dict[str, float] = {}
        cols = st.columns(2)
        for index, feature in enumerate(features):
            item = schema["features"].get(feature, {})
            label = feature_label(feature, schema)
            help_text = item.get("help", "")
            target_col = cols[index % 2]
            if item.get("kind") == "binary":
                choice = target_col.selectbox(
                    label,
                    options=[0, 1],
                    format_func=lambda x: "Yes" if int(x) == 1 else "No",
                    index=int(item.get("default", 0)),
                    help=help_text,
                )
                values[feature] = float(choice)
            else:
                values[feature] = float(
                    target_col.number_input(
                        label,
                        value=float(item.get("default", calculator_metadata.get("continuous_medians", {}).get(feature, 0.0))),
                        step=float(item.get("step", 1.0)),
                        format=item.get("format", "%.2f"),
                        help=help_text,
                    )
                )
        submitted = st.form_submit_button("Estimate Probability")

    if not submitted:
        return

    frame = pd.DataFrame([values], columns=features)
    result = predict_dataframe(model, frame, threshold)
    probability = float(result.loc[0, "predicted_probability"])
    predicted_class = int(result.loc[0, "predicted_class"])
    status_text = "Likely conversion success" if predicted_class else "Lower predicted conversion success"
    status_class = "status-high" if predicted_class else "status-low"

    st.markdown('<div class="section-label">Model Output</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-value">{probability:.1%}</div>
            <div class="result-label">Estimated probability of SVT conversion success</div>
            <div class="{status_class}">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns(3)
    left.metric("Probability", f"{probability:.1%}")
    middle.metric("Decision threshold", f"{threshold:.1%}")
    right.metric("Predicted class", "Success" if predicted_class else "Lower success")

    st.progress(min(max(probability, 0.0), 1.0))
    with st.expander("Input and prediction values"):
        st.dataframe(display_dataframe(result), use_container_width=True, hide_index=True)
    render_shap_force_plot(model, frame)


def render_batch_prediction(model: Any, calculator_metadata: dict[str, Any]) -> None:
    features = list(calculator_metadata["selected_features"])
    threshold = float(calculator_metadata.get("best_threshold", 0.5))
    st.markdown('<div class="section-label">Batch Prediction</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Batch CSV", type=["csv"])
    if uploaded is None:
        st.download_button(
            "Download sample CSV",
            data=(APP_DIR / "sample_input.csv").read_bytes(),
            file_name="sample_input.csv",
            mime="text/csv",
        )
        return

    try:
        raw = pd.read_csv(uploaded)
        frame = coerce_model_frame(raw, features)
        result = predict_dataframe(model, frame, threshold)
    except Exception as exc:
        st.error(str(exc))
        return

    st.dataframe(display_dataframe(result), use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="svt_conversion_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="SVT Conversion Success Prediction", layout="centered")
    inject_style()
    try:
        with st.spinner("Loading prediction model..."):
            model, calculator_metadata, model_metadata, schema = load_model_bundle()
    except Exception as exc:
        st.error("The prediction model could not be loaded. Please check deployment files and package versions.")
        with st.expander("Startup technical details"):
            st.write(str(exc))
        st.stop()

    st.markdown(
        """
        <div class="model-hero">
            <div class="model-title">SVT Conversion Success Prediction Model</div>
            <p class="model-subtitle">
                Online calculator for estimating the probability of successful supraventricular tachycardia conversion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Model Summary")
        st.write(f"Algorithm: `{calculator_metadata.get('best_model', model_metadata.get('best_model', 'Model'))}`")
        st.write("Preprocessing: `MissForest imputation`")
        st.write(f"Threshold: `{float(calculator_metadata.get('best_threshold', 0.5)):.3f}`")
        st.write(f"Outcome: `{calculator_metadata.get('target', 'SVT conversion success')}`")
        st.write("Predictors")
        for predictor in calculator_metadata["selected_features"]:
            st.write(f"- {display_name(predictor)}")
        st.divider()
        st.caption("Research-use online model. Clinical application requires professional review and local validation.")

    tab_single, tab_batch = st.tabs(["Single prediction", "Batch CSV"])
    with tab_single:
        render_single_prediction(model, calculator_metadata, model_metadata, schema)
    with tab_batch:
        render_batch_prediction(model, calculator_metadata)


if __name__ == "__main__":
    main()
