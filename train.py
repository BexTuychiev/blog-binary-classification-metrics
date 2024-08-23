import os
import sys
import neptune
import pandas as pd
import xgboost as xgb
import lightgbm
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

model = lightgbm.LGBMClassifier(**MODEL_PARAMS)
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
)

# Calculate metrics
accuracy = accuracy_score(y_test, y_test_preds)
roc_auc = roc_auc_score(y_test, y_test_probs[:, 1])  # Assuming binary classification
precision = precision_score(y_test, y_test_preds, average="weighted")
recall = recall_score(y_test, y_test_preds, average="weighted")
f1 = f1_score(y_test, y_test_preds, average="weighted")
pr_auc = average_precision_score(
    y_test, y_test_probs[:, 1], average="weighted"
)  # PR AUC

# Log metrics to Neptune
run["accuracy"] = accuracy
run["roc_auc"] = roc_auc
run["precision"] = precision
run["recall"] = recall
run["f1"] = f1
run["pr_auc"] = pr_auc

run.stop()
