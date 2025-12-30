from sklearn.base import BaseEstimator, TransformerMixin
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class ProcessFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_duplicates=False, verbose=True):
        self.drop_duplicates = drop_duplicates
        self.verbose = verbose

    def fit(self, X, y=None):
        df = X.copy()

        logger.info(f"Fit feature engineering trên {len(df):,} dòng")

        df["is_root"] = (df["userId"] == 0).astype(int)
        df["is_child_process"] = (df["parentProcessId"] != 1).astype(int)
        df["return_is_error"] = (df["returnValue"] < 0).astype(int)

        self.mnt_freq_ = df.groupby("mountNamespace").size()
        self.mnt_unique_process_ = df.groupby("mountNamespace")["processId"].nunique()

        self.user_freq_ = df.groupby("userId").size()
        self.user_unique_process_ = df.groupby("userId")["processId"].nunique()
        self.user_root_ratio_ = df.groupby("userId")["is_root"].mean()
        self.user_error_ratio_ = df.groupby("userId")["return_is_error"].mean()

        self.proc_freq_ = df.groupby("processId").size()
        self.proc_unique_parent_ = df.groupby("processId")["parentProcessId"].nunique()
        self.proc_root_ratio_ = df.groupby("processId")["is_root"].mean()
        self.proc_child_ratio_ = df.groupby("processId")["is_child_process"].mean()
        self.proc_error_ratio_ = df.groupby("processId")["return_is_error"].mean()
        self.proc_avg_args_ = df.groupby("processId")["argsNum"].mean()

        self.parent_freq_ = df.groupby("parentProcessId").size()
        self.parent_unique_child_ = df.groupby("parentProcessId")["processId"].nunique()
        self.parent_root_child_ratio_ = df.groupby("parentProcessId")["is_root"].mean()
        self.parent_error_ratio_ = df.groupby("parentProcessId")["return_is_error"].mean()

        self.global_freq_ = 1
        self.global_ratio_ = df["return_is_error"].mean()
        self.global_args_ = df["argsNum"].mean()

        logger.info("Hoàn tất fit feature engineering")
        return self

    def transform(self, X):
        df = X.copy()

        logger.info(f"Transform dữ liệu đầu vào: {len(df):,} dòng")

        if self.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            logger.info(f"Loại bỏ trùng lặp: {before - len(df):,} dòng")

        if "threadId" in df.columns:
            df["same_process_threadId"] = (df["processId"] == df["threadId"]).astype(int)
            df.drop(columns="threadId", inplace=True, errors="ignore")

        df["is_root"] = (df["userId"] == 0).astype(int)
        df["is_child_process"] = (df["parentProcessId"] != 1).astype(int)
        df["return_is_error"] = (df["returnValue"] < 0).astype(int)
        df["return_is_zero"] = (df["returnValue"] == 0).astype(int)
        df["return_is_positive"] = (df["returnValue"] > 0).astype(int)

        df["mnt_freq"] = df["mountNamespace"].map(self.mnt_freq_).fillna(self.global_freq_)
        df["mnt_unique_process"] = df["mountNamespace"].map(self.mnt_unique_process_).fillna(self.global_freq_)

        df["user_freq"] = df["userId"].map(self.user_freq_).fillna(self.global_freq_)
        df["user_unique_process"] = df["userId"].map(self.user_unique_process_).fillna(self.global_freq_)
        df["user_root_ratio"] = df["userId"].map(self.user_root_ratio_).fillna(self.global_ratio_)
        df["user_error_ratio"] = df["userId"].map(self.user_error_ratio_).fillna(self.global_ratio_)

        df["proc_freq"] = df["processId"].map(self.proc_freq_).fillna(self.global_freq_)
        df["proc_unique_parent"] = df["processId"].map(self.proc_unique_parent_).fillna(self.global_freq_)
        df["proc_root_ratio"] = df["processId"].map(self.proc_root_ratio_).fillna(self.global_ratio_)
        df["proc_child_ratio"] = df["processId"].map(self.proc_child_ratio_).fillna(self.global_ratio_)
        df["proc_error_ratio"] = df["processId"].map(self.proc_error_ratio_).fillna(self.global_ratio_)
        df["proc_avg_args"] = df["processId"].map(self.proc_avg_args_).fillna(self.global_args_)

        df["parent_freq"] = df["parentProcessId"].map(self.parent_freq_).fillna(self.global_freq_)
        df["parent_unique_child"] = df["parentProcessId"].map(self.parent_unique_child_).fillna(self.global_freq_)
        df["parent_root_child_ratio"] = df["parentProcessId"].map(self.parent_root_child_ratio_).fillna(self.global_ratio_)
        df["parent_error_ratio"] = df["parentProcessId"].map(self.parent_error_ratio_).fillna(self.global_ratio_)

        logger.info(f"Hoàn tất transform, kích thước cuối: {df.shape}")
        return df
