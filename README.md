# Water Monitoring Agent (SDG 6: Clean Water and Sanitation)

An agentic AI system with a machine-learning model. It reads water-quality
readings, checks them against safe limits, predicts a risk score with a
Random Forest model, decides SAFE / CAUTION / UNSAFE, gives advice, raises
alerts and keeps a history.

Agent loop: observe -> analyze -> decide -> act -> learn

## Files
| File | What it does |
|---|---|
| config.py | Settings: safe limits, file paths |
| generate_data.py | Makes the (synthetic) training dataset |
| train_model.py | Trains and saves the Random Forest model, writes accuracy results |
| agent.py | The agent: observe, analyze, decide, act, learn |
| app.py | The Streamlit web page |
| test_agent.py | Self-test (should print "All tests passed.") |
| sample_readings.csv | Example data to upload |
| requirements.txt | Packages needed |
| setup_windows.bat | One-click setup for Windows |
| run_windows.bat | One-click start for Windows |

## Run on Windows (easiest)
1. Install Python 3.11 or newer from python.org. Tick "Add Python to PATH".
2. Unzip. Open the folder until you can see app.py and setup_windows.bat.
3. Double-click setup_windows.bat. Wait until it says "Setup complete".
4. Double-click run_windows.bat. The browser opens the app.

## Run in VS Code
1. File > Open Folder, choose the folder that contains app.py.
2. Terminal > New Terminal. Then run, one line at a time:
   python -m venv venv
   venv\Scripts\python -m pip install -r requirements.txt
   venv\Scripts\python generate_data.py
   venv\Scripts\python train_model.py
   venv\Scripts\python test_agent.py
   venv\Scripts\python -m streamlit run app.py
   (Mac/Linux: use python3 and venv/bin/python)
   Using venv\Scripts\python avoids the PowerShell "scripts are disabled" error.

## Deploy
Push to GitHub, then deploy app.py on Streamlit Community Cloud. The model
trains itself the first time the app starts.

## Optional email alerts
Set the environment variables ALERT_SMTP_HOST, ALERT_SMTP_USER,
ALERT_SMTP_PASSWORD and ALERT_TO. Without them, alerts are written to alerts.log.

## Important notes
- The training data is synthetic. Replace data/water_quality.csv with real lab data
  (columns: ph, turbidity, tds, chlorine, potable) and run train_model.py again.
- Safe limits follow BIS IS 10500 acceptable limits. Check the latest standard.
- This is a study project, not a certified testing tool.
