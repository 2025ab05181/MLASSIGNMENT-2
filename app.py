import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff
from pathlib import Path


# Path to the directory where the script lives
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("🧠 ML Classification Models - Bank Dataset")
st.markdown("Upload test **bank.csv**, select model, view metrics & predictions.")

# Load models & preprocessors
@st.cache_resource
def load_all():
    models = {
        'Logistic Regression': joblib.load('model/logistic_model.pkl'),
        'Decision Tree': joblib.load('model/dt_model.pkl'),
        'KNN': joblib.load('model/knn_model.pkl'),
        'Naive Bayes': joblib.load('model/nb_model.pkl'),
        'Random Forest': joblib.load('model/rf_model.pkl'),
        'XGBoost': joblib.load('model/xgb_model.pkl')
    }
    full_preproc = joblib.load('model/preprocessor.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
    tree_models = ['Decision Tree', 'Random Forest', 'XGBoost']
    return models, full_preproc, label_encoders, tree_models

models, full_preproc, label_encoders, tree_models = load_all()

# Sidebar
st.sidebar.header("⚙️ Controls")
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

# File upload
uploaded_file = st.file_uploader("📁 Upload Test bank.csv", type="csv")
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {test_df.shape[0]} rows × {test_df.shape[1]} cols")
    st.dataframe(test_df.head(), use_container_width=True)
    
    target_col = test_df.columns[-1]
    X_test = test_df.iloc[:, :-1]
    y_test_true = pd.factorize(test_df[target_col])[0]  # Auto-encode target
    
    cat_cols = list(label_encoders.keys())
    
    if st.button("🚀 Evaluate Model", type="primary"):
        try:
            # Model-specific preprocessing
            if selected_model_name in tree_models:
                # Trees: Label encode cats (16 feats)
                X_test_tree = X_test.copy()
                for col in cat_cols:
                    if col in X_test_tree.columns:
                        X_test_tree[col] = label_encoders[col].transform(X_test_tree[col].astype(str))
                X_test_processed = X_test_tree.values.astype(float)
            else:
                # Others: Full OneHot+Scale (42 feats)
                X_test_processed = full_preproc.transform(X_test)
            
            # Predict
            y_pred = selected_model.predict(X_test_processed)
            y_pred_proba = selected_model.predict_proba(X_test_processed)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test_true, y_pred)
            auc = roc_auc_score(y_test_true, y_pred_proba)
            mcc = matthews_corrcoef(y_test_true, y_pred)
            report = classification_report(y_test_true, y_pred, output_dict=True)
            
            # Layout: Metrics + Visuals
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC-ROC", f"{auc:.4f}")
            col3.metric("MCC", f"{mcc:.4f}")
            col4.metric("Precision", f"{report['weighted avg']['precision']:.4f}")
            col5.metric("F1-Score", f"{report['weighted avg']['f1-score']:.4f}")
            
            # Classification Report Table
            st.subheader("📊 Classification Report")
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
            
            # Confusion Matrix
            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y_test_true, y_pred)
            fig = ff.create_annotated_heatmap(
                z=cm, x=['No', 'Yes'], y=['No', 'Yes'], 
                colorscale='Blues', showscale=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions
            test_df['Predicted'] = ['Yes' if p == 1 else 'No' for p in y_pred]
            test_df['Probability'] = y_pred_proba
            st.subheader("🎯 Predictions")
            st.dataframe(test_df[['Predicted', 'Probability']].tail(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Ensure preprocessors/models match dataset. Rerun training scripts.")
else:
    st.info("👆 Upload **bank.csv** to start (must match training data columns).")

# Footer
st.markdown("---")
st.caption("💻 BITS Pilani ML Assignment 2 | Deployed on Streamlit Cloud")
