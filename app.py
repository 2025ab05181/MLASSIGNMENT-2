import streamlit as st
st.set_page_config(page_title="ML Assignment 2", layout="wide")

import sys
from pathlib import Path

import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
import plotly.figure_factory as ff


BASE_DIR = Path(__file__).resolve().parent

st.title("🧠 ML Classification Models - Bank Dataset")
st.markdown("Upload test **bank.csv**, select model, view metrics & predictions.")
st.sidebar.subheader("⬇️ Download sample dataset")

with open("bank.csv", "rb") as f:
    st.sidebar.download_button(
        label="Download bank.csv",
        data=f,
        file_name="bank.csv",
        mime="text/csv",
    )


# Optional debug (AFTER set_page_config)
with st.expander("🔧 Environment info", expanded=False):
    st.write(sys.version)
    import sklearn
    st.write("sklearn", sklearn.__version__)
    st.write("joblib", joblib.__version__)


@st.cache_resource
def load_all():
    models = {
        "Logistic Regression": joblib.load("model/logistic_model.pkl"),
        "Decision Tree": joblib.load("model/dt_model.pkl"),
        "kNN": joblib.load("model/knn_model.pkl"),           # show as kNN (consistent with README)
        "Naive Bayes": joblib.load("model/nb_model.pkl"),
        "Random Forest (Ensemble)": joblib.load("model/rf_model.pkl"),
        "XGBoost (Ensemble)": joblib.load("model/xgb_model.pkl"),
    }
    full_preproc = joblib.load("model/preprocessor.pkl")
    label_encoders = joblib.load("model/label_encoders.pkl")

    tree_models = {"Decision Tree", "Random Forest (Ensemble)", "XGBoost (Ensemble)"}
    return models, full_preproc, label_encoders, tree_models


models, full_preproc, label_encoders, tree_models = load_all()

# Sidebar
st.sidebar.header("⚙️ Controls")
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

uploaded_file = st.file_uploader("📁 Upload Test bank.csv", type="csv")

if uploaded_file is None:
    st.info("👆 Upload **bank.csv** to start (must match training data columns).")
    st.markdown("---")
    st.caption("💻 BITS Pilani ML Assignment 2 | Deployed on Streamlit Cloud")
    st.stop()

test_df = pd.read_csv(uploaded_file)
st.success(f"✅ Loaded {test_df.shape[0]} rows × {test_df.shape[1]} cols")
st.dataframe(test_df.head(), use_container_width=True)

# Split features/target
target_col = test_df.columns[-1]
X_test = test_df.iloc[:, :-1].copy()

# Stable target encoding: yes->1, no->0
y_str = test_df[target_col].astype(str).str.strip().str.lower()
y_test_true = (y_str == "yes").astype(int).values

# Basic checks
with st.expander("🧪 Debug checks", expanded=False):
    st.write("Target column:", target_col)
    st.write("Unique target values:", sorted(y_str.unique().tolist()))
    st.write("y_test_true counts:", pd.Series(y_test_true).value_counts().to_dict())
    st.write("Selected model:", selected_model_name)

cat_cols = list(label_encoders.keys())

if st.button("🚀 Evaluate Model", type="primary"):
    try:
        # Preprocess according to model type
        if selected_model_name in tree_models:
            X_test_tree = X_test.copy()
            for col in cat_cols:
                if col in X_test_tree.columns:
                    # IMPORTANT: must match training encoding
                    X_test_tree[col] = label_encoders[col].transform(X_test_tree[col].astype(str))
            X_test_processed = X_test_tree.values.astype(float)
        else:
            X_test_processed = full_preproc.transform(X_test)

        with st.expander("📐 Processed shape", expanded=False):
            st.write("X_test_processed shape:", getattr(X_test_processed, "shape", None))

        # Predict
        y_pred = selected_model.predict(X_test_processed)

        # Predict proba safely (pick proba column for class 1 based on classes_)
        y_pred_proba = None
        if hasattr(selected_model, "predict_proba"):
            proba = selected_model.predict_proba(X_test_processed)

            # classes_ gives the order of columns in predict_proba
            classes = getattr(selected_model, "classes_", None)
            with st.expander("🔎 Probability debug", expanded=False):
                st.write("Model classes_:", classes)

            if classes is not None and 1 in list(classes):
                pos_idx = list(classes).index(1)
            else:
                # fallback: assume column 1 is positive
                pos_idx = 1 if proba.shape[1] > 1 else 0

            y_pred_proba = proba[:, pos_idx]
        else:
            # if model has no proba, make a dummy score
            y_pred_proba = y_pred.astype(float)

        # Metrics
        acc = accuracy_score(y_test_true, y_pred)

        # AUC is undefined if only one class present in y_true
        if len(np.unique(y_test_true)) < 2:
            auc = float("nan")
            st.warning("AUC-ROC is undefined because the uploaded file contains only one class in the target.")
        else:
            auc = roc_auc_score(y_test_true, y_pred_proba)

        mcc = matthews_corrcoef(y_test_true, y_pred)
        report = classification_report(y_test_true, y_pred, output_dict=True, zero_division=0)

        # Layout: Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC-ROC", "N/A" if np.isnan(auc) else f"{auc:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")
        col4.metric("Precision", f"{report['weighted avg']['precision']:.4f}")
        col5.metric("F1-Score", f"{report['weighted avg']['f1-score']:.4f}")

        # Classification report
        st.subheader("📊 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test_true, y_pred)
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=["No", "Yes"],
            y=["No", "Yes"],
            colorscale="Blues",
            showscale=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Predictions
        out_df = test_df.copy()
        out_df["Predicted"] = np.where(y_pred == 1, "Yes", "No")
        out_df["Probability"] = y_pred_proba

        st.subheader("🎯 Predictions (last 10)")
        st.dataframe(out_df[["Predicted", "Probability"]].tail(10), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Ensure preprocessors/models match dataset columns and that label encoders cover categorical values.")

st.markdown("---")
st.caption("💻 BITS Pilani ML Assignment 2 | Deployed on Streamlit Cloud")
