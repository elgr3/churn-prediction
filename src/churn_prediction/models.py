import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn_prediction.config import SEED
from churn_prediction.features import build_preprocessor


def _with_preprocessing(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def candidate_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": _with_preprocessing(
            LogisticRegression(class_weight="balanced", max_iter=2000)
        ),
        "random_forest": _with_preprocessing(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )
        ),
        "xgboost": _with_preprocessing(
            XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=4.0,
                tree_method="hist",
                random_state=SEED,
                n_jobs=-1,
            )
        ),
    }


def compare(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = {}
    for name, pipe in models.items():
        res = cross_validate(pipe, X, y, cv=cv, scoring=["f1", "roc_auc"])
        rows[name] = {
            "f1_mean": round(float(res["test_f1"].mean()), 4),
            "f1_std": round(float(res["test_f1"].std()), 4),
            "auc_mean": round(float(res["test_roc_auc"].mean()), 4),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("f1_mean", ascending=False)


def test_metrics(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "f1": round(float(f1_score(y_test, pred)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, proba)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
    }
