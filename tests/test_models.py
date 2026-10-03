from churn_prediction import config
from churn_prediction.models import candidate_models, compare
from churn_prediction.models import test_metrics as evaluate_on_test
from tests.test_data import fake_frame


def _xy(n=600):
    df = fake_frame(n, seed=3)
    df["Exited"] = ((df["Age"] > 50) & (df["IsActiveMember"] == 0)).astype(int)
    return df.drop(columns=config.ID_COLUMNS + [config.TARGET]), df[config.TARGET]


def test_three_candidates():
    assert set(candidate_models()) == {"logistic_regression", "random_forest", "xgboost"}


def test_compare_ranks_by_f1():
    X, y = _xy()
    table = compare(candidate_models(), X, y)
    assert list(table.columns) == ["f1_mean", "f1_std", "auc_mean"]
    assert table["f1_mean"].is_monotonic_decreasing


def test_test_metrics_keys():
    X, y = _xy()
    pipe = candidate_models()["random_forest"].fit(X, y)
    m = evaluate_on_test(pipe, X, y)
    assert set(m) == {"f1", "accuracy", "auc_roc", "precision", "recall"}
    assert m["auc_roc"] > 0.9
