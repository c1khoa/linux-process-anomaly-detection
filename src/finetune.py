from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
from optuna.samplers import TPESampler
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import time
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
    average_precision_score
)
import logging
from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNED_SAVE_PATH = config.TUNED_SAVE_PATH
METADATA_SAVE_PATH = config.METADATA_SAVE_PATH


def tune_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    logger.info("Tune Logistic Regression")

    model = LogisticRegression(
        max_iter=500,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }

    grid = GridSearchCV(
        model,
        param_grid,
        scoring='average_precision',
        cv=3,
        n_jobs=-1
    )

    start = time.time()
    grid.fit(X_train_scaled, y_train)
    elapsed = time.time() - start

    logger.info(f"Hoàn tất Logistic Regression | Thời gian: {elapsed:.2f}s")
    return grid.best_estimator_, elapsed


def tune_random_forest(X_train_scaled, y_train, X_test_scaled, y_test, n_iter=5):
    logger.info("Tune Random Forest")

    model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    )

    param_dist = {
        'n_estimators': [200, 500, 800],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='average_precision',
        cv=3,
        n_jobs=-1,
        random_state=42
    )

    start = time.time()
    search.fit(X_train_scaled, y_train)
    elapsed = time.time() - start

    logger.info(f"Hoàn tất Random Forest | Thời gian: {elapsed:.2f}s")
    return search.best_estimator_, elapsed


def tune_xgboost(X_train_scaled, y_train, X_test_scaled, y_test):
    logger.info("Tune XGBoost")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'tree_method': 'hist',
            'n_estimators': 500,
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'verbosity': 0
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train_scaled, y_train)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        return average_precision_score(y_test, y_proba)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))

    start = time.time()
    study.optimize(objective, n_trials=50)
    elapsed = time.time() - start

    model = xgb.XGBClassifier(
        n_estimators=500,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        **study.best_params
    )
    model.fit(X_train_scaled, y_train)

    logger.info(f"Hoàn tất XGBoost | Thời gian: {elapsed:.2f}s")
    return model, elapsed


def tune_lightgbm(X_train_scaled, y_train, X_test_scaled, y_test):
    logger.info("Tune LightGBM")

    def objective(trial):
        params = {
            'n_estimators': 500,
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_scaled, y_train)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        return average_precision_score(y_test, y_proba)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))

    start = time.time()
    study.optimize(objective, n_trials=50)
    elapsed = time.time() - start

    model = lgb.LGBMClassifier(
        n_estimators=500,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        **study.best_params
    )
    model.fit(X_train_scaled, y_train)

    logger.info(f"Hoàn tất LightGBM | Thời gian: {elapsed:.2f}s")
    return model, elapsed


def find_best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.05, 0.95, 100)
    best_t, best_score = 0.5, -1

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_score:
            best_score, best_t = score, t

    return best_t, best_score


def finetune_models(X_train_scaled, y_train, X_test_scaled, y_test):
    logger.info("Bắt đầu fine-tune models")

    tuned_models = {}
    tuned_times = {}

    tuned_models['logistic_regression'], tuned_times['logistic_regression'] = tune_logistic_regression(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    tuned_models['random_forest'], tuned_times['random_forest'] = tune_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    tuned_models['xgboost'], tuned_times['xgboost'] = tune_xgboost(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    tuned_models['lightgbm'], tuned_times['lightgbm'] = tune_lightgbm(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    tuned_results = {}

    for name, model in tuned_models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        best_t, best_f1 = find_best_threshold(y_test, y_proba)
        y_pred = (y_proba >= best_t).astype(int)

        tuned_results[name] = {
            'model': model,
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'accuracy': accuracy_score(y_test, y_pred),
            'recall_class_1': classification_report(
                y_test, y_pred, output_dict=True
            ).get('1', {}).get('recall', 0.0),
            'best_threshold': best_t,
            'f1_at_best_threshold': best_f1,
            'train_time_sec': tuned_times.get(name)
        }

        logger.info(f"{name} | F1: {tuned_results[name]['f1_score']:.4f}")

    os.makedirs(TUNED_SAVE_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, result in tuned_results.items():
        path = os.path.join(TUNED_SAVE_PATH, f"{name}_tuned.pkl")
        if isinstance(result['model'], xgb.XGBModel):
            result['model'].get_booster().save_model(path.replace('.pkl', '.json'))
        else:
            joblib.dump(result['model'], path)

    metadata = {
        'timestamp': timestamp,
        'optimized_metric': 'f1',
        'models': {
            name: {
                'f1_score': r['f1_score'],
                'accuracy': r['accuracy'],
                'recall_class_1': r['recall_class_1'],
                'best_threshold': r['best_threshold'],
                'f1_at_best_threshold': r['f1_at_best_threshold'],
                'train_time_sec': r['train_time_sec']
            }
            for name, r in tuned_results.items()
        }
    }

    os.makedirs(METADATA_SAVE_PATH, exist_ok=True)
    joblib.dump(metadata, f"{METADATA_SAVE_PATH}/metadata_{timestamp}.pkl")
    with open(f"{METADATA_SAVE_PATH}/metadata_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info("Hoàn tất fine-tune và lưu kết quả")
