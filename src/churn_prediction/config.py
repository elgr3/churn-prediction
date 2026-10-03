from pathlib import Path

SEED = 42
OPENML_ID = 45586
TARGET = "Exited"
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

CATEGORICAL = ["Geography", "Gender", "age_band"]
NUMERIC = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "balance_zero",
    "balance_salary_ratio",
    "products_x_active",
]

AGE_BINS = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]
MIN_SEGMENT_SIZE = 100

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
