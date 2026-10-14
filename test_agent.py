"""Quick self-test. Run:  python test_agent.py"""
import pandas as pd

from agent import get_model, run_agent, validate
from config import METRICS_FILE
import json


def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    if not condition:
        raise SystemExit(1)


good = {"location": "A", "ph": 7.2, "turbidity": 0.4, "tds": 300, "chlorine": 0.5}
one_bad = {"location": "B", "ph": 7.2, "turbidity": 1.3, "tds": 300, "chlorine": 0.5}
bad = {"location": "C", "ph": 5.8, "turbidity": 5.0, "tds": 700, "chlorine": 0.0}

get_model()
r = run_agent([good, one_bad, bad], save=False)
check("clean water is SAFE", r[0]["status"] == "SAFE")
check("one problem is CAUTION or UNSAFE", r[1]["status"] in ("CAUTION", "UNSAFE"))
check("many problems is UNSAFE", r[2]["status"] == "UNSAFE")
check("risk score is higher for bad water", r[2]["ml_risk"] > r[0]["ml_risk"])

metrics = json.loads(METRICS_FILE.read_text())
check("model accuracy is above 80%", metrics["accuracy"] > 0.80)

try:
    validate(pd.DataFrame({"ph": [7.0]}))
    check("missing columns are rejected", False)
except ValueError:
    check("missing columns are rejected", True)

print("\nAll tests passed.")
