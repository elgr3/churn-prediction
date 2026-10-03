import numpy as np
import pandas as pd

from churn_prediction import config
from churn_prediction.data import load_dataset, split


def fake_frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "RowNumber": np.arange(n),
            "CustomerId": rng.integers(1e7, 2e7, n),
            "Surname": ["X"] * n,
            "CreditScore": rng.integers(350, 850, n),
            "Geography": rng.choice(["France", "Spain", "Germany"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 92, n),
            "Tenure": rng.integers(0, 11, n),
            "Balance": np.where(rng.random(n) < 0.35, 0.0, rng.random(n) * 2e5),
            "NumOfProducts": rng.integers(1, 5, n),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": rng.integers(0, 2, n),
            "EstimatedSalary": rng.random(n) * 2e5,
            "Exited": (rng.random(n) < 0.2).astype(int),
        }
    )


def test_load_drops_identifiers(tmp_path):
    fake_frame().to_parquet(tmp_path / "bank_churn.parquet")
    X, y = load_dataset(cache_dir=tmp_path)
    assert not set(config.ID_COLUMNS) & set(X.columns)
    assert config.TARGET not in X.columns and set(y.unique()) <= {0, 1}


def test_split_stratified(tmp_path):
    fake_frame(1000).to_parquet(tmp_path / "bank_churn.parquet")
    X_tr, X_te, y_tr, y_te = split(*load_dataset(cache_dir=tmp_path))
    assert len(X_te) == 200 and abs(y_tr.mean() - y_te.mean()) < 0.03
