import json

import pandas as pd
import streamlit as st

from agent import get_model, load_history, run_agent, validate
from config import FEATURES, LIMITS, METRICS_FILE

st.set_page_config(page_title="Water Monitoring Agent", page_icon="💧")
st.title("Water Monitoring Agent")
st.caption("SDG 6: Clean Water and Sanitation  |  Agentic AI for water quality monitoring")

SHOW = {"SAFE": st.success, "CAUTION": st.warning, "UNSAFE": st.error}

tab1, tab2, tab3, tab4 = st.tabs(["Check one reading", "Upload a CSV", "History", "Model"])

# ---------------------------------------------------------------- tab 1
with tab1:
    location = st.text_input("Location", "School tank")
    c1, c2 = st.columns(2)
    ph = c1.number_input("pH", min_value=0.0, max_value=14.0, value=7.2, step=0.1)
    turbidity = c2.number_input("Turbidity (NTU)", min_value=0.0, value=0.5, step=0.1)
    tds = c1.number_input("TDS (mg/L)", min_value=0.0, value=300.0, step=10.0)
    chlorine = c2.number_input("Chlorine (mg/L)", min_value=0.0, value=0.4, step=0.1)

    if st.button("Check water"):
        reading = {"location": location, "ph": ph, "turbidity": turbidity,
                   "tds": tds, "chlorine": chlorine}
        result = run_agent([reading])[0]
        SHOW[result["status"]](result["message"])
        st.metric("AI risk score", f"{result['ml_risk']:.0%}")

    st.caption("Safe ranges used: " + ", ".join(
        f"{k} {lo} to {hi}" for k, (lo, hi) in LIMITS.items()))

# ---------------------------------------------------------------- tab 2
with tab2:
    st.write("Upload a CSV with the columns: location, " + ", ".join(FEATURES))
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        try:
            data = validate(pd.read_csv(file))
            results = run_agent(data.to_dict("records"))
            counts = pd.Series([r["status"] for r in results]).value_counts()
            st.write(f"Checked {len(results)} locations: " +
                     ", ".join(f"{n} {k}" for k, n in counts.items()))
            for r in results:
                SHOW[r["status"]](f"{r['location']}: {r['message']}")
        except Exception as error:
            st.error(f"Could not read this file. {error}")

# ---------------------------------------------------------------- tab 3
with tab3:
    history = load_history()
    if history.empty:
        st.info("No checks yet. Run a check and it will appear here.")
    else:
        st.subheader("Results so far")
        st.bar_chart(history["status"].value_counts())
        st.dataframe(history)

# ---------------------------------------------------------------- tab 4
with tab4:
    get_model()   # trains the model the first time if it does not exist
    if METRICS_FILE.exists():
        m = json.loads(METRICS_FILE.read_text())
        st.write(f"Random Forest classifier, trained on {m['rows_train']} rows and "
                 f"tested on {m['rows_test']} unseen rows.")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{m['accuracy']:.1%}")
        b.metric("Precision", f"{m['precision']:.1%}")
        c.metric("Recall", f"{m['recall']:.1%}")
        d.metric("F1 score", f"{m['f1']:.1%}")
        st.subheader("Confusion matrix")
        cm = pd.DataFrame(m["confusion_matrix"],
                          index=["Actually unsafe", "Actually safe"],
                          columns=["Predicted unsafe", "Predicted safe"])
        st.dataframe(cm)
        st.subheader("Which readings matter most")
        st.bar_chart(pd.Series(m["feature_importance"]))
        st.caption("Training data is synthetic (made by generate_data.py). "
                   "Use real lab data for real-world decisions.")
