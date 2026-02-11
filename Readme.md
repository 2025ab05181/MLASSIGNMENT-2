# Step 5: README.md (Model Comparison)

## a. Problem statement
Predict whether a bank customer will subscribe to a **term deposit** (`yes`/`no`) based on customer profile and marketing campaign attributes from `bank.csv`.

---

## b. Dataset description 
| Item | Description |
|---|---|
| Dataset file | `bank.csv` |
| Task type | Binary classification (`yes`/`no`) |
| Rows × Columns | 4521 × 17 (16 features + 1 target) |
| Target column | Last column in the CSV (`y`) |
| Class balance | Imbalanced (majority class much larger than minority) |

---

## c. Models used 
Metrics were computed on the **saved trained models** (`.pkl`) using a stratified 80/20 split with `random_state=42`.

### 1) Comparison table 
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.826691 | 0.910291 | 0.832507 | 0.793951 | 0.812772 | 0.652258 |
| Decision Tree | 0.975817 | 0.976833 | 0.954710 | 0.996219 | 0.975023 | 0.952416 |
| kNN | 0.868786 | 0.941622 | 0.861873 | 0.861059 | 0.861466 | 0.736838 |
| Naive Bayes | 0.721451 | 0.815184 | 0.812321 | 0.535917 | 0.645786 | 0.457176 |
| Random Forest (Ensemble) | 0.977161 | 0.994633 | 0.958144 | 0.995274 | 0.976356 | 0.954929 |
| XGBoost (Ensemble) | 0.939991 | 0.981946 | 0.915468 | 0.962193 | 0.938249 | 0.880988 |

### 2) Observations on model performance 
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good baseline with strong AUC, but lower than tree ensembles, suggesting nonlinear feature interactions exist. |
| Decision Tree | Very high scores on this split; may overfit compared to ensembles but performs strongly here. |
| kNN | Better than Logistic Regression in Accuracy/F1/MCC, but still behind the best ensemble models. |
| Naive Bayes | Lowest overall; recall is relatively low, which reduces F1 and MCC. |
| Random Forest (Ensemble) | Best overall across most metrics, showing robust performance on this dataset. |
| XGBoost (Ensemble) | Very strong performance and close to Random Forest, confirming boosted trees are effective on tabular data. |
