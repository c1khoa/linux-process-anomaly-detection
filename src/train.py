from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)
from src import config
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


DATA_PATH = config.DATA_PATH
PREPROCESSING_PATH = config.PREPROCESSING_PATH
BASELINE_SAVE_PATH = config.BASELINE_SAVE_PATH
IMAGE_SAVE_PATH = config.IMAGE_SAVE_PATH

os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
os.makedirs(BASELINE_SAVE_PATH, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name, figsize=(6, 5)):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}/cfs_mtx_{model_name}.png")
    plt.close()

def compute_metrics(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, output_dict=True)
    recall_class_1 = report.get('1', {}).get('recall', 0.0)
    acc = accuracy_score(y_true, y_pred)
    return f1_macro, recall_class_1, acc

def train(X_train_scaled, y_train, X_test_scaled, y_test):
    logger.info("Bắt đầu huấn luyện các mô hình baseline")
    logger.info(
        f"Kích thước tập train: {X_train_scaled.shape}, "
        f"tập test: {X_test_scaled.shape}"
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    baseline_models = {
        'logistic_regression': LogisticRegression(
            max_iter=500,
            solver='lbfgs',
            class_weight='balanced',
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=42,
            class_weight='balanced'
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist"
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary',
            class_weight={0: 1, 1: scale_pos_weight},
            random_state=42,
            verbose=-1
        )
    }
    baseline_results = {}

    for name, model in baseline_models.items():
        logger.info(f"Huấn luyện mô hình: {name}")

        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test_scaled)

        f1_macro, recall_class_1, acc = compute_metrics(y_test, y_pred)

        logger.info(
            f"{name} | F1-macro={f1_macro:.4f} | "
            f"Accuracy={acc:.4f} | "
            f"Recall lớp bất thường={recall_class_1:.4f} | "
            f"Thời gian train={train_time:.2f}s"
        )

        baseline_results[name] = {
            'model': model,
            'f1_score': f1_macro,
            'recall_class_1': recall_class_1,
            'accuracy': acc,
            'train_time_sec': train_time
        }

        plot_confusion_matrix(y_test, y_pred, name)

        logger.info("Lưu các mô hình baseline")

    for name, result in baseline_results.items():
        filepath = os.path.join(BASELINE_SAVE_PATH, f"{name}_baseline.pkl")

        if isinstance(result['model'], xgb.XGBModel):
            save_path = filepath.replace('.pkl', '.json')
            result['model'].get_booster().save_model(save_path)
            logger.info(f"Đã lưu mô hình XGBoost: {save_path}")
        else:
            joblib.dump(result['model'], filepath)
            logger.info(f"Đã lưu mô hình: {filepath}")

    logger.info("Hoàn tất huấn luyện baseline")



