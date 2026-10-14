"""Settings shared by every file in the project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "water_quality.csv"
MODEL_FILE = BASE_DIR / "model" / "water_model.joblib"
METRICS_FILE = BASE_DIR / "model" / "metrics.json"
HISTORY_FILE = BASE_DIR / "history.csv"
ALERT_LOG = BASE_DIR / "alerts.log"

# The four water-quality readings the agent uses.
FEATURES = ["ph", "turbidity", "tds", "chlorine"]

# Safe range (low, high) for each reading, based on BIS IS 10500 acceptable
# limits. Check the latest standard before any real-world use.
LIMITS = {
    "ph": (6.5, 8.5),
    "turbidity": (0.0, 1.0),    # NTU
    "tds": (0.0, 500.0),        # mg/L
    "chlorine": (0.2, 5.0),     # mg/L, free residual chlorine
}

NAMES = {
    "ph": "pH",
    "turbidity": "Turbidity (NTU)",
    "tds": "TDS (mg/L)",
    "chlorine": "Chlorine (mg/L)",
}
