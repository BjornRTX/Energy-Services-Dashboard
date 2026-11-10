# Energy Dashboard: Central IST Building

Interactive dashboard (Python, Plotly Dash) for the hourly electricity consumption of the Central Building at Instituto Superior Técnico, Lisbon. Made for the Energy Services course (project 2).

It shows the measured data for 2019, a next-hour forecast against the measured power, the forecast error metrics (MAE, RMSE, cvRMSE, NMBE and more), a feature selection tool, and the 2017–2018 training data.

## How to open it (Windows)

1. Install Python 3.12 from python.org (tick "Add python.exe to PATH" in the installer). Python 3.10 to 3.12 works; newer versions may not.
2. Unzip this folder.
3. Double-click `start-dashboard.bat`. The first start installs the packages (about two minutes) and needs internet. Then your browser opens the dashboard.

If the browser does not open by itself, go to http://127.0.0.1:8050 once the window says "Running on". Close the black window to stop the dashboard.

## How to open it by hand (any system)

```
python -m venv .venv
.venv\Scripts\activate          (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python local-dashboard.py
```

Then open http://127.0.0.1:8050.

## Files

| File | Purpose |
|---|---|
| `local-dashboard.py` | The dashboard (both models) |
| `online-dashboard.py` | Same code, used for the online version |
| `requirements.txt` | Exact package versions; the saved models need scikit-learn 1.6.1 and NumPy 2.0.2 |
| `regression_data_2017-2018.csv`, `prepared_test_data_2019.csv` | Prepared data (training years and test year) |
| `neural_networks_model.pkl`, `best_regression_model_113309-home.pkl` | Saved models |
| `assets/first style.css` | Page design |

## Notes

- The forecast uses the measured power of the previous hour as one input, which is how a next-hour forecast works.
- With fewer than all four inputs ticked, the dashboard trains a simple Random Forest on 2017–2018 with only those inputs, because the saved models need all four.
- The online version (Render) offers the neural network only, because the 64 MB Random Forest file is too large for the repository.

Björn Siemss, student number 113309.
