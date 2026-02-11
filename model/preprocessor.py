import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# 
df = pd.read_csv('bank.csv')  # Your uploaded file
print("Your dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Dtypes:\n", df.dtypes)

# Auto-detect: last column = target, rest = features
target_col = df.columns[-1]
X = df.iloc[:, :-1]  # All but last column
y = (df[target_col] == 'yes').astype(int) if 'yes' in df[target_col].unique() else pd.factorize(df[target_col])[0]

print(f"Target: {target_col}, Unique: {df[target_col].unique()}")

# Auto-identify categoricals/numerics
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
unknown_cols = set(X.columns) - set(cat_cols) - set(num_cols)
if unknown_cols:
    print("Warning: Unknown dtypes:", unknown_cols)
    num_cols.extend(unknown_cols)  # Treat as numeric

print(f"Cats ({len(cat_cols)}): {cat_cols}")
print(f"Nums ({len(num_cols)}): {num_cols}")

# 1. FULL PREPROCESSOR: OneHot cats + Scale nums
full_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough'
)

full_preprocessor.fit(X)
joblib.dump(full_preprocessor, 'preprocessor.pkl')
print("✓ preprocessor.pkl created")

# 2. TREE PREPROCESSOR: LabelEncode cats only
label_encoders = {}
X_tree = X.copy()
for col in cat_cols:
    le = LabelEncoder()
    X_tree[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

joblib.dump(label_encoders, 'label_encoders.pkl')
print("✓ label_encoders.pkl created")

print(f"Ready! Full feats: {full_preprocessor.transform(X).shape[1]}, Tree feats: {X_tree.shape[1]}")
