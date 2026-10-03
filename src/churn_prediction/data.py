from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from churn_prediction.config import DATA_DIR, ID_COLUMNS, OPENML_ID, SEED, TARGET

CACHE_NAME = "bank_churn.parquet"


def load_dataset(cache_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    cache = cache_dir / CACHE_NAME
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
        df.to_parquet(cache)
    y = df[TARGET].astype(float).astype(int).rename(TARGET)
    X = df.drop(columns=[c for c in ID_COLUMNS + [TARGET] if c in df.columns])
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
