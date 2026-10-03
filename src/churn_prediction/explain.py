from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def explain_xgboost(pipe: Pipeline, X_sample: pd.DataFrame, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    prep = pipe.named_steps["prep"]
    names = list(prep.named_steps["encode"].get_feature_names_out())
    Xt = pd.DataFrame(prep.transform(X_sample), columns=names)
    explanation = shap.TreeExplainer(pipe.named_steps["model"])(Xt)

    shap.summary_plot(explanation, Xt, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=120)
    plt.close()

    ranked = [names[i] for i in np.argsort(np.abs(explanation.values).mean(axis=0))[::-1]]
    for feature in ranked[:3]:
        shap.plots.scatter(explanation[:, feature], show=False)
        plt.tight_layout()
        plt.savefig(figures_dir / f"shap_dependence_{feature}.png", dpi=120)
        plt.close()
    return ranked
