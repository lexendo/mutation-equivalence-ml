import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score

# =========================
# CONFIG
# =========================
TRAIN_CSV = "mutantbench_train.csv"
TEST_CSV = "mutantbench_test.csv"


print("Loading TRAIN data...")
train_df = pd.read_csv(TRAIN_CSV)

train_df["before"] = train_df["before"].fillna("")
train_df["after"] = train_df["after"].fillna("")

train_df["code_diff"] = "- " + train_df["before"] + " + " + train_df["after"]

X_train = train_df["code_diff"]
y_train = train_df["label"].astype(int)


clf = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95
    )),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("Training improved text-based model on full training set...")
clf.fit(X_train, y_train)

print("Loading TEST data...")
test_df = pd.read_csv(TEST_CSV)

test_df["before"] = test_df["before"].fillna("")
test_df["after"] = test_df["after"].fillna("")

test_df["code_diff"] = "- " + test_df["before"] + " + " + test_df["after"]

X_test = test_df["code_diff"]
y_test = test_df["label"].astype(int)


print("Evaluating on test set...")

proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("\nROC AUC:", roc_auc_score(y_test, proba))
print("\nClassification report:\n")
print(classification_report(y_test, pred, digits=4))
