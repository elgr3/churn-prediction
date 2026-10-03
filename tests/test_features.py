import numpy as np

from churn_prediction import config
from churn_prediction.features import add_features, build_preprocessor
from tests.test_data import fake_frame


def _X(n=50):
    return fake_frame(n).drop(columns=config.ID_COLUMNS + [config.TARGET])


def test_derived_columns():
    X = _X().assign(Balance=[0.0] + [100.0] * 49, EstimatedSalary=[0.0] + [50.0] * 49)
    out = add_features(X)
    assert out.loc[0, "balance_zero"] == 1 and out.loc[1, "balance_zero"] == 0
    assert np.isfinite(out["balance_salary_ratio"]).all()
    assert out.loc[1, "balance_salary_ratio"] == 100.0 / 51.0


def test_extreme_ages_get_a_band():
    out = add_features(_X(2).assign(Age=[17, 95]))
    assert list(out["age_band"]) == ["<30", "60+"]


def test_add_features_does_not_mutate_input():
    X = _X()
    add_features(X)
    assert "age_band" not in X.columns


def test_preprocessor_ignores_unknown_geography_and_names_features():
    pre = build_preprocessor().fit(_X(200))
    out = pre.transform(_X(1).assign(Geography="Italy"))
    names = list(pre.named_steps["encode"].get_feature_names_out())
    assert out.shape[1] == len(names)
    assert "Geography_France" in names and "Age" in names
