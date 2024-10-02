import os
import neptune
import pandas as pd
import xgboost as xgb  # type: ignore
import matplotlib.pyplot as plt
import argparse

from dotenv import load_dotenv
from neptune.integrations.xgboost import NeptuneCallback

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="XGBoost model training with Neptune logging"
)
parser.add_argument("name", type=str, help="Name for the Neptune run")
parser.add_argument("learning_rate", type=float, help="Learning rate for XGBoost")
parser.add_argument("n_estimators", type=int, help="Number of estimators for XGBoost")
args = parser.parse_args()

# Load the environment variables
load_dotenv()

# Download and read the data
TRAIN_PATH = "https://raw.githubusercontent.com/neptune-ai/blog-binary-classification-metrics/master/data/train.csv"
TEST_PATH = "https://raw.githubusercontent.com/neptune-ai/blog-binary-classification-metrics/master/data/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Prepare the data for training
feature_names = [col for col in train.columns if col not in ["isFraud"]]

X_train, y_train = train[feature_names], train["isFraud"]
X_test, y_test = test[feature_names], test["isFraud"]

# Start experiment before training
project_name = os.getenv("NEPTUNE_PROJECT_NAME")
api_token = os.getenv("NEPTUNE_API_TOKEN")

run = neptune.init_run(project=project_name, api_token=api_token, name=args.name)

# Train model
MODEL_PARAMS = {
    "random_state": 1234,
    "learning_rate": args.learning_rate,
    "n_estimators": args.n_estimators,
}

model = xgb.XGBClassifier(**MODEL_PARAMS)
model.fit(X_train, y_train)

# Evaluate model
y_test_probs = model.predict_proba(X_test)
y_test_preds = model.predict(X_test)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    brier_score_loss,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)

# Calculate metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_test_preds).ravel()

# Confusion matrix derived metrics
false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (tp + fn)
true_negative_rate = tn / (tn + fp)
negative_predictive_value = tn / (tn + fn)
false_discovery_rate = fp / (tp + fp)

# Basic classification metrics
accuracy = accuracy_score(y_test, y_test_preds)
precision = precision_score(y_test, y_test_preds)
recall = recall_score(y_test, y_test_preds)
f1score = f1_score(y_test, y_test_preds)

# Advanced classification metrics
fbeta = fbeta_score(y_test, y_test_preds, beta=1)
f2 = fbeta_score(y_test, y_test_preds, beta=2)
cohen_kappa = cohen_kappa_score(y_test, y_test_preds)
matthews_corr = matthews_corrcoef(y_test, y_test_preds)

# Probability-based metrics
roc_auc = roc_auc_score(y_test, y_test_probs[:, 1])
avg_precision = average_precision_score(y_test, y_test_probs[:, 1])
loss = log_loss(y_test, y_test_probs)
brier = brier_score_loss(y_test, y_test_probs[:, 1])

# Log metrics to Neptune
run["false_positive_rate"] = false_positive_rate
run["false_negative_rate"] = false_negative_rate
run["true_negative_rate"] = true_negative_rate
run["negative_predictive_value"] = negative_predictive_value
run["false_discovery_rate"] = false_discovery_rate
run["accuracy"] = accuracy
run["precision"] = precision
run["recall"] = recall
run["f1score"] = f1score
run["fbeta"] = fbeta
run["f2"] = f2
run["cohen_kappa"] = cohen_kappa
run["matthews_corr"] = matthews_corr
run["roc_auc"] = roc_auc
run["avg_precision"] = avg_precision
run["loss"] = loss
run["brier"] = brier

# Generate and log plots
plot_functions = [
    ("roc_curve", RocCurveDisplay),
    ("precision_recall_curve", PrecisionRecallDisplay),
]

for plot_name, plot_class in plot_functions:
    fig, ax = plt.subplots()
    plot_class.from_predictions(y_test, y_test_probs[:, 1], ax=ax)
    run[f"images/{plot_name}_fig"].upload(neptune.types.File.as_image(fig))
    plt.close(fig)

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_test_preds, ax=ax)
run["images/confusion_matrix_fig"].upload(neptune.types.File.as_image(fig))
plt.close(fig)

# Log model parameters to Neptune
run["learning_rate"] = args.learning_rate
run["n_estimators"] = args.n_estimators

run.stop()
