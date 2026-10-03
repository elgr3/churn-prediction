import pandas as pd

from churn_prediction.config import MIN_SEGMENT_SIZE
from churn_prediction.features import age_band

KEYS = ["Geography", "age_band", "NumOfProducts"]


def risk_segments(
    X: pd.DataFrame, y: pd.Series, min_size: int = MIN_SEGMENT_SIZE, top: int = 5
) -> pd.DataFrame:
    df = X[["Geography", "Age", "NumOfProducts"]].copy()
    df["age_band"] = age_band(df["Age"])
    df["churn"] = y.to_numpy()
    seg = (
        df.groupby(KEYS, observed=True)["churn"]
        .agg(customers="size", churn_rate="mean")
        .reset_index()
    )
    seg = seg[seg["customers"] >= min_size].copy()
    seg["churn_rate"] = seg["churn_rate"].round(4)
    seg["lift"] = (seg["churn_rate"] / y.mean()).round(2)
    return seg.sort_values("churn_rate", ascending=False).head(top).reset_index(drop=True)
