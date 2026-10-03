import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn_prediction.config import AGE_BINS, AGE_LABELS, CATEGORICAL, NUMERIC


def age_band(age: pd.Series) -> pd.Series:
    """Map ages to AGE_LABELS; out-of-range ages are clipped into the edge bands."""
    return pd.cut(
        age.clip(lower=AGE_BINS[0] + 1, upper=AGE_BINS[-1] - 1),
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    ).astype(str)


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out["balance_zero"] = (out["Balance"] == 0).astype(int)
    out["balance_salary_ratio"] = out["Balance"] / (out["EstimatedSalary"] + 1.0)
    out["balance_salary_ratio"] = out["balance_salary_ratio"].replace([np.inf, -np.inf], 0.0)
    out["products_x_active"] = (out["NumOfProducts"] * out["IsActiveMember"]).astype(int)
    out["age_band"] = age_band(out["Age"])
    return out


def build_preprocessor() -> Pipeline:
    encode = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", FunctionTransformer(add_features)), ("encode", encode)])
