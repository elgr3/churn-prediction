"""Compare models in CV, evaluate the best on test, explain XGBoost, find risk segments."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from churn_prediction import config  # noqa: E402
from churn_prediction.data import load_dataset, split  # noqa: E402
from churn_prediction.explain import explain_xgboost  # noqa: E402
from churn_prediction.models import candidate_models, compare  # noqa: E402
from churn_prediction.models import test_metrics as evaluate_on_test  # noqa: E402
from churn_prediction.segments import risk_segments  # noqa: E402


def main() -> None:
    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = split(X, y)

    models = candidate_models()
    table = compare(models, X_tr, y_tr)
    best_name = table.index[0]
    fitted = {name: pipe.fit(X_tr, y_tr) for name, pipe in models.items()}

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ax = table["f1_mean"].plot.barh(xerr=table["f1_std"], figsize=(6, 3), title="CV F1 by model")
    ax.figure.tight_layout()
    ax.figure.savefig(config.FIGURES_DIR / "model_comparison.png", dpi=120)
    plt.close(ax.figure)

    ranked = explain_xgboost(
        fitted["xgboost"], X_te.sample(1000, random_state=config.SEED), config.FIGURES_DIR
    )
    segments = risk_segments(X, y)
    segments.to_csv(config.REPORTS_DIR / "segments.csv", index=False)

    report = {
        "dataset": {"n_rows": int(len(X)), "churn_rate": round(float(y.mean()), 4)},
        "cv_comparison": table.to_dict(orient="index"),
        "selected_model": best_name,
        "test": evaluate_on_test(fitted[best_name], X_te, y_te),
        "xgboost_test": evaluate_on_test(fitted["xgboost"], X_te, y_te),
        "top_features": ranked[:5],
        "risk_segments": segments.to_dict(orient="records"),
    }
    out = config.REPORTS_DIR / "metrics.json"
    out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
