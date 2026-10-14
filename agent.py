"""Water Monitoring Agent: observe -> analyze -> decide -> act -> learn."""
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

import joblib
import pandas as pd

from config import ALERT_LOG, FEATURES, HISTORY_FILE, LIMITS, MODEL_FILE, NAMES

ADVICE = {
    ("ph", "low"): "Water is acidic. Check pipes and the source, then retest.",
    ("ph", "high"): "Water is alkaline. Check the source, then retest.",
    ("turbidity", "high"): "Water is cloudy and may hide germs. Filter and boil it.",
    ("tds", "high"): "Many dissolved solids. Avoid regular drinking and retest.",
    ("chlorine", "low"): "Not enough chlorine to kill germs. Disinfect the tank.",
    ("chlorine", "high"): "Too much chlorine. Let the water stand and inform the supervisor.",
}

_model = None


def get_model():
    """Load the trained model. If it is missing or unreadable, train it."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_FILE)["model"]
        except Exception:
            from train_model import train
            train(verbose=False)
            _model = joblib.load(MODEL_FILE)["model"]
    return _model


def validate(df):
    """Check that an uploaded table has the columns the agent needs."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(missing) +
                         ". Needed: location, " + ", ".join(FEATURES))
    df = df.copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[FEATURES].isna().any().any():
        raise ValueError("Some readings are empty or not numbers. Fix the CSV and try again.")
    if "location" not in df.columns:
        df["location"] = [f"Row {i + 1}" for i in range(len(df))]
    return df


# ---- Step 1: OBSERVE -------------------------------------------------------
def observe(path):
    """Read water readings from a CSV file."""
    return validate(pd.read_csv(path)).to_dict("records")


# ---- Step 2: ANALYZE -------------------------------------------------------
def analyze(reading):
    """Rule check (safe limits) plus the ML risk score (0 to 1)."""
    problems = []
    for name, (low, high) in LIMITS.items():
        value = reading[name]
        if value < low:
            problems.append((name, value, "low"))
        elif value > high:
            problems.append((name, value, "high"))

    model = get_model()
    row = pd.DataFrame([[reading[f] for f in FEATURES]], columns=FEATURES)
    unsafe_col = list(model.classes_).index(0)
    ml_risk = float(model.predict_proba(row)[0][unsafe_col])
    return problems, ml_risk


# ---- Step 3: DECIDE --------------------------------------------------------
def decide(problems, ml_risk):
    """Combine the rules and the ML score into SAFE, CAUTION or UNSAFE."""
    if not problems:
        status = "SAFE"
    elif len(problems) == 1:
        status = "CAUTION"
    else:
        status = "UNSAFE"
    if status == "SAFE" and ml_risk >= 0.6:
        status = "CAUTION"          # rules pass but the model sees a risky pattern
    elif status == "CAUTION" and ml_risk >= 0.8:
        status = "UNSAFE"
    return status


# ---- Step 4: ACT -----------------------------------------------------------
def act(reading, status, problems, ml_risk, save=True):
    """Write the advice, raise an alert if needed, and save the event."""
    risk_text = f"AI risk score {ml_risk:.0%}."
    if problems:
        details = "; ".join(f"{NAMES[n]} is {d} ({v})" for n, v, d in problems)
        steps = " ".join(ADVICE.get((n, d), "Retest the water.") for n, _, d in problems)
        message = f"{status}: {details}. {steps}"
        if status == "UNSAFE":
            message += " Do not drink without boiling for at least one minute."
        message += " " + risk_text
    elif status == "CAUTION":
        message = (f"CAUTION: readings are inside the limits, but the model sees a risky "
                   f"pattern. {risk_text} Retest soon.")
    else:
        message = f"SAFE: all readings are within safe limits. {risk_text} Keep monitoring."

    if save:
        if status != "SAFE":
            send_alert(reading, status, message)
        save_event(reading, status, ml_risk, message)    # Step 5: LEARN
    return message


def send_alert(reading, status, message):
    """Write an alert to alerts.log. Email is sent only if you set the
    ALERT_SMTP_HOST, ALERT_SMTP_USER, ALERT_SMTP_PASSWORD and ALERT_TO
    environment variables (optional; never put passwords in the code)."""
    line = f"{datetime.now():%Y-%m-%d %H:%M} | {reading['location']} | {message}\n"
    with open(ALERT_LOG, "a", encoding="utf-8") as f:
        f.write(line)

    host, user = os.getenv("ALERT_SMTP_HOST"), os.getenv("ALERT_SMTP_USER")
    password, to = os.getenv("ALERT_SMTP_PASSWORD"), os.getenv("ALERT_TO")
    if not all([host, user, password, to]):
        return
    try:
        mail = EmailMessage()
        mail["Subject"] = f"Water alert: {status} at {reading['location']}"
        mail["From"], mail["To"] = user, to
        mail.set_content(message)
        with smtplib.SMTP_SSL(host, 465, timeout=15) as server:
            server.login(user, password)
            server.send_message(mail)
    except Exception as error:     # an email problem must never stop the agent
        print("Email alert failed:", error)


# ---- Step 5: LEARN ---------------------------------------------------------
def save_event(reading, status, ml_risk, message):
    """Keep a history of every check so trends can be studied."""
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "location": reading["location"],
           **{f: reading[f] for f in FEATURES},
           "ml_risk": round(ml_risk, 3), "status": status, "message": message}
    pd.DataFrame([row]).to_csv(HISTORY_FILE, mode="a", index=False,
                               header=not HISTORY_FILE.exists())


def load_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(HISTORY_FILE, on_bad_lines="skip")


# ---- The full agent loop ---------------------------------------------------
def run_agent(readings, save=True):
    results = []
    for reading in readings:
        reading = dict(reading)
        reading.setdefault("location", "Unknown")
        problems, ml_risk = analyze(reading)
        status = decide(problems, ml_risk)
        message = act(reading, status, problems, ml_risk, save=save)
        results.append({"location": reading["location"], "status": status,
                        "ml_risk": ml_risk, "message": message})
    return results


if __name__ == "__main__":
    from config import BASE_DIR
    for r in run_agent(observe(BASE_DIR / "sample_readings.csv")):
        print(f"{r['location']:<20} {r['status']:<8} risk {r['ml_risk']:.0%}")
        print("   ", r["message"])
