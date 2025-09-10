import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import pickle

# Load
df = pd.read_csv("heart.csv")

# Assume target
target_col = "HeartDisease"
X = df.drop(columns=[target_col])
y = df[target_col]

# Feature types
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Pipeline components
numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Pipelines
logreg_pipe = Pipeline([("preprocessor", preprocessor),
                        ("classifier", LogisticRegression(max_iter=1000, solver="liblinear"))])
rf_pipe = Pipeline([("preprocessor", preprocessor),
                    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Fit
logreg_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)

# Predict & probs
y_pred_lr = logreg_pipe.predict(X_test)
y_proba_lr = logreg_pipe.predict_proba(X_test)[:,1]

y_pred_rf = rf_pipe.predict(X_test)
y_proba_rf = rf_pipe.predict_proba(X_test)[:,1]

# Metrics
lr_acc = accuracy_score(y_test, y_pred_lr)
rf_acc = accuracy_score(y_test, y_pred_rf)
lr_auc = roc_auc_score(y_test, y_proba_lr)
rf_auc = roc_auc_score(y_test, y_proba_rf)

print("LogReg acc:", lr_acc, "AUC:", lr_auc)
print("RandomForest acc:", rf_acc, "AUC:", rf_auc)
print("LogReg report:\n", classification_report(y_test, y_pred_lr))
print("RF report:\n", classification_report(y_test, y_pred_rf))

# Save best model
best_model = rf_pipe if rf_acc >= lr_acc else logreg_pipe
pickle.dump(best_model, open("best_model.pkl", "wb"))

# Save test results for inspection
test_results = X_test.copy()
test_results["true"] = y_test.values
test_results["pred_rf"] = y_pred_rf
test_results["pred_lr"] = y_pred_lr
test_results["proba_rf"] = y_proba_rf
test_results["proba_lr"] = y_proba_lr
test_results.to_csv("test_results.csv", index=False)

# Optional: plot ROC
plt.figure(figsize=(8,6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
plt.plot(fpr_lr, tpr_lr, label=f"LogReg AUC={lr_auc:.3f}")
plt.plot(fpr_rf, tpr_rf, label=f"RF AUC={rf_auc:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
