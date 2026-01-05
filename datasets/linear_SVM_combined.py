import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score


TRAIN_CSV = "mutantbench_train.csv"
TEST_CSV = "mutantbench_test.csv"


print("Loading TRAIN data...")
train_df = pd.read_csv(TRAIN_CSV)

train_df["operator"] = train_df["operator"].fillna("UNKNOWN")
train_df["before"] = train_df["before"].fillna("")
train_df["after"] = train_df["after"].fillna("")

train_df["code_diff"] = "- " + train_df["before"] + " + " + train_df["after"]

X_train = train_df[["operator", "code_diff"]]
y_train = train_df["label"].astype(int)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "operator",
            OneHotEncoder(handle_unknown="ignore"),
            ["operator"],
        ),
        (
            "code",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=3,
                max_df=0.95,
            ),
            "code_diff",
        ),
    ]
)


svm = LinearSVC(
    class_weight="balanced",
    C=1.1,
)

clf = Pipeline(
    steps=[
        ("features", preprocessor),
        ("model", CalibratedClassifierCV(svm)),
    ]
)

print("Training combined operator + text model (Linear SVC)...")
clf.fit(X_train, y_train)


print("Loading TEST data...")
test_df = pd.read_csv(TEST_CSV)

test_df["operator"] = test_df["operator"].fillna("UNKNOWN")
test_df["before"] = test_df["before"].fillna("")
test_df["after"] = test_df["after"].fillna("")

test_df["code_diff"] = "- " + test_df["before"] + " + " + test_df["after"]

X_test = test_df[["operator", "code_diff"]]
y_test = test_df["label"].astype(int)


print("Evaluating on test set...")

proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("\nROC AUC:", roc_auc_score(y_test, proba))
print("\nClassification report:\n")
print(classification_report(y_test, pred, digits=4))
