import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


TRAIN_CSV = "mutantbench_train.csv"
TEST_CSV = "mutantbench_test.csv"


print("Loading TRAIN data...")
train_df = pd.read_csv(TRAIN_CSV)

train_df["operator"] = train_df["operator"].fillna("UNKNOWN")

X_train = train_df[["operator"]]
y_train = train_df["label"].astype(int)

clf = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ("model", LogisticRegression(max_iter=1000))
])

print("Training operator-only model on full training set...")
clf.fit(X_train, y_train)

print("Loading TEST data...")
test_df = pd.read_csv(TEST_CSV)

test_df["operator"] = test_df["operator"].fillna("UNKNOWN")

X_test = test_df[["operator"]]
y_test = test_df["label"].astype(int)


print("Evaluating on test set...")

proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("\nROC AUC:", roc_auc_score(y_test, proba))
print("\nClassification report:\n")
print(classification_report(y_test, pred, digits=4))
