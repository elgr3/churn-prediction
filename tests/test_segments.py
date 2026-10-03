import pandas as pd

from churn_prediction.segments import risk_segments


def _frame():
    rows = []
    rows += [{"Geography": "Germany", "Age": 55, "NumOfProducts": 1, "y": 1}] * 150
    rows += [{"Geography": "Germany", "Age": 55, "NumOfProducts": 2, "y": 0}] * 150
    rows += [{"Geography": "Spain", "Age": 25, "NumOfProducts": 2, "y": 1}] * 10  # tiny
    rows += [{"Geography": "France", "Age": 35, "NumOfProducts": 2, "y": 0}] * 300
    df = pd.DataFrame(rows)
    return df.drop(columns="y"), df["y"]


def test_small_segments_are_excluded():
    X, y = _frame()
    seg = risk_segments(X, y, min_size=100)
    assert not (seg["Geography"] == "Spain").any()


def test_highest_risk_first_with_lift():
    X, y = _frame()
    seg = risk_segments(X, y, min_size=100)
    first = seg.iloc[0]
    assert first["Geography"] == "Germany" and first["NumOfProducts"] == 1
    assert first["churn_rate"] == 1.0
    assert first["lift"] == round(1.0 / y.mean(), 2)
