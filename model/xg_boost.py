
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *


from pathlib import Path


# Path to the directory where the script lives
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv('bank.csv')
X = df.iloc[:, :-1]
y = (df.iloc[:, -1] == 'yes').astype(int)

label_encoders = joblib.load(str(BASE_DIR)+'/label_encoders.pkl')
cat_cols = list(label_encoders.keys())
X_tree = X.copy()
for col in cat_cols:
    if col in X_tree.columns:
        X_tree[col] = label_encoders[col].transform(X[col].astype(str))
X_processed = X_tree.values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)
joblib.dump(model, str(BASE_DIR)+'/xgb_model.pkl')

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_proba),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred)
}

print("=== XGBOOST ===")
for k, v in metrics.items(): print(f"{k}: {v:.4f}")
print(f"README: | XGBoost | {' | '.join(f'{v:.4f}' for v in metrics.values())} |")
print("✅ xgb_model.pkl saved!")
