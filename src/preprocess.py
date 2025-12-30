import os
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from src.features import ProcessFeatureEngineer
from src import config
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


DATA_PATH = config.DATA_PATH
PREPROCESSING_PATH = config.PREPROCESSING_PATH

def run_preprocessing():
    if not os.path.exists(DATA_PATH):
        return

    df = pd.read_csv(DATA_PATH)

    feature_engineer = ProcessFeatureEngineer(verbose=True, drop_duplicates=False)
    df_processed = feature_engineer.fit_transform(df)

    X = df_processed.drop(columns=['target'])
    y = df_processed['target']
    groups = df_processed['processId']

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    train_hash = pd.util.hash_pandas_object(X_train, index=False)
    test_hash = pd.util.hash_pandas_object(X_test, index=False)
    _ = set(train_hash) & set(test_hash)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(PREPROCESSING_PATH, exist_ok=True)
    joblib.dump(feature_engineer, f'{PREPROCESSING_PATH}/feature_engineer.pkl')
    joblib.dump(scaler, f'{PREPROCESSING_PATH}/scaler.pkl')
    
    return X_train_scaled, y_train, X_test_scaled, y_test
