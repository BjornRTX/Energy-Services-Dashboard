"""Energy Dashboard for the Central IST Building (Energy Services, Project 2).

Put this file in one folder together with
    regression_data_2017-2018.csv
    prepared_test_data_2019.csv
    neural_networks_model.pkl
    best_regression_model_113309-home.pkl      (optional: without it the Random Forest is shown as unavailable)
    assets/                                    (style sheet)

Run locally:   python local-dashboard.py   and open http://127.0.0.1:8050
Deployment:    the same file works on Render with  gunicorn <module name>:server
"""
import os
import traceback
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# --------------------------------------------------------------------------------------
# Files: everything is expected next to this script
# --------------------------------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # e.g. code pasted into a console
    BASE_DIR = Path(os.getcwd())

df_2017_2018 = pd.read_csv(BASE_DIR / "regression_data_2017-2018.csv")
df_2019 = pd.read_csv(BASE_DIR / "prepared_test_data_2019.csv")

MODEL_PATHS = {
    "regression": BASE_DIR / "best_regression_model_113309-home.pkl",
    "neural": BASE_DIR / "neural_networks_model.pkl",
}
MODEL_NAMES = {"regression": "Random Forest", "neural": "Neural Network"}
AVAILABLE_MODELS = [key for key, path in MODEL_PATHS.items() if path.exists()]


def allow_newer_numpy_pickles():
    """Models saved with a newer NumPy store their random generator in a form that older NumPy versions
    reject ("... is not a known BitGenerator module", "state is not a legacy MT19937 state").
    The generator is not needed for predictions, so older NumPy is taught to accept it and to ignore
    a saved state it cannot read. Does nothing when NumPy already understands the file."""
    try:
        import numpy.random._pickle as rp

        old_bit_generator = getattr(rp, "__bit_generator_ctor")
    except (ImportError, AttributeError):
        return

    class TolerantRandomState(np.random.RandomState):
        def __setstate__(self, state):
            try:
                super().__setstate__(state)
            except Exception:  # state written by another NumPy version; irrelevant for predictions
                pass

    def as_bit_generator(name):
        """Class, ready-made object, or the old string name -> BitGenerator object."""
        if isinstance(name, np.random.BitGenerator):
            return name
        if isinstance(name, type):
            return name()
        return old_bit_generator(name)

    def bit_generator_ctor(name="MT19937"):
        return as_bit_generator(name)

    def randomstate_ctor(name="MT19937", bit_generator_ctor_=None):
        return TolerantRandomState(as_bit_generator(name))

    setattr(rp, "__bit_generator_ctor", bit_generator_ctor)
    setattr(rp, "__randomstate_ctor", randomstate_ctor)


@lru_cache(maxsize=None)
def load_model(key):
    """Each model file is read once, not at every click."""
    allow_newer_numpy_pickles()
    return joblib.load(MODEL_PATHS[key])


# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------
def preprocess_df(df):
    df.rename(columns={"Solar Radiation in W/m^2": "Solar Radiation in W/m²"}, inplace=True)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


df_2017_2018 = preprocess_df(df_2017_2018)
df_2019 = preprocess_df(df_2019)

df_2017 = df_2017_2018[df_2017_2018["Date"].dt.year == 2017]
df_2018 = df_2017_2018[df_2017_2018["Date"].dt.year == 2018]
DATA_BY_YEAR = {2017: df_2017, 2018: df_2018, 2019: df_2019}

features = ["Last Hour Power in kW", "Temperature in °C", "Humidity in %", "Solar Radiation in W/m²"]
weather_options = [{"label": f, "value": f} for f in features[1:]]


@lru_cache(maxsize=None)
def train_quick_model(used):
    """The saved models need all four inputs. For a smaller set of inputs a simple Random Forest is trained
    on the 2017-2018 data with only those inputs (takes about two seconds, then it is kept in memory)."""
    X = df_2017_2018[list(used)]
    y = df_2017_2018["Power in kW"]
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X, y)

# name, unit, factor for display (cvRMSE and NMBE are stored as fractions and shown in %)
METRIC_INFO = {
    "MAE": ("Mean Absolute Error", "kW", 1),
    "MBE": ("Mean Bias Error", "kW", 1),
    "MSE": ("Mean Squared Error", "kW²", 1),
    "RMSE": ("Root Mean Squared Error", "kW", 1),
    "cvRMSE": ("Coefficient of Variation of RMSE", "%", 100),
    "NMBE": ("Normalized Mean Bias Error", "%", 100),
}

# one plain sentence per metric, shown on its tile
METRIC_HINT = {
    "MAE": "Average error per hour. Lower is better.",
    "MBE": "Average error with sign: actual minus prediction.",
    "MSE": "Average squared error. Big misses count a lot.",
    "RMSE": "Like MAE, but big misses count more.",
    "cvRMSE": "RMSE as a share of the average power.",
    "NMBE": "Bias as a share of the average power (actual minus prediction).",
}

BLUE, ORANGE, GREY = "#0b6bcb", "#e8731a", "#5b6b7f"
FONT = "Segoe UI, Helvetica, Arial, sans-serif"

# --------------------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------------------
def card(title, subtitle, *children):
    head = [html.H2(title, className="card-title")]
    if subtitle:
        head.append(html.P(subtitle, className="card-sub"))
    return html.Section(head + list(children), className="card")


def notice(text, kind="info"):
    return html.Div(text, className=f"notice {kind}")


def control(label, component):
    return html.Div([html.Label(label, className="field-label"), component], className="control")


def style_figure(fig, title, y_title, x_title=None, height=420):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=18)),
        height=height,
        margin=dict(l=64, r=24, t=64, b=90),
        font=dict(family=FONT, size=13, color="#1f2d3d"),
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, yanchor="top", x=0.5, xanchor="center", title_text=""),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e8edf3", zeroline=False)
    return fig


def month_axis(fig):
    fig.update_xaxes(dtick="M1", tickformat="%b")
    return fig


def graph(fig):
    return html.Div(dcc.Graph(figure=fig, config={"displaylogo": False}, style={"width": "100%"}), className="graph-card")


# --------------------------------------------------------------------------------------
# App and layout
# --------------------------------------------------------------------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Energy Dashboard"
server = app.server  # needed for deployment with gunicorn

TABS = [
    {"label": "Forecast and metrics", "value": "forecast"},
    {"label": "Raw data 2019", "value": "raw"},
    {"label": "Feature selection", "value": "features"},
    {"label": "Training data 2017–2018", "value": "training"},
]

app.layout = html.Div(
    [
        html.Header(
            html.Div(
                [
                    html.H1("Energy Dashboard"),
                    html.P("Central IST Building · hourly electricity consumption and next-hour forecast", className="topbar-sub"),
                ],
                className="topbar-inner",
            ),
            className="topbar",
        ),
        html.Main(
            [
                dcc.RadioItems(id="main-tab", options=TABS, value="forecast", className="tabbar", labelClassName="tab"),
                html.Div(id="content-container"),
            ],
            className="container",
        ),
        html.Footer("Energy Services · Project 2 · Técnico Lisboa · Björn Siemss (113309)", className="footer"),
    ],
    className="app",
)


def choice_group(component_id, options, value, multi=False):
    cls = dcc.Checklist if multi else dcc.RadioItems
    return cls(id=component_id, options=options, value=value, className="choice-group", labelClassName="choice")


def weather_selector(component_id):
    return control("Weather variable", choice_group(component_id, weather_options, weather_options[0]["value"]))


def data_section(prefix, selector_id):
    """Power consumption first, then the weather variable choice with its graph (and last hour power for 2019)."""
    return html.Div(
        [
            html.Div(id=f"{prefix}-power-graph"),
            html.Div(weather_selector(selector_id), className="after-graph"),
            html.Div(id=f"{prefix}-weather-graph"),
            html.Div(id=f"{prefix}-last-hour-graph"),
        ],
        className="graph-stack",
    )


def explainer():
    """Three short boxes: what the forecast does, how it works, how to read the chart."""
    steps = [
        ("What it does", "Predicts how much electricity the Central Building will use in the next hour, for every hour from 1 January to 11 April 2019."),
        ("How it works", "A trained model receives the power of the last hour plus temperature, humidity and solar radiation, and returns the expected power. It learned from 2017 and 2018 and has never seen 2019."),
        ("How to read it", "Blue is the measured power, orange is the prediction. The closer the two lines, the better. The tiles below give the average error in kW."),
    ]
    return html.Section(
        [html.Div([html.H3(title), html.P(text)], className="step") for title, text in steps],
        className="card explainer",
    )


def forecast_tab():
    default_model = "neural" if "neural" in AVAILABLE_MODELS else (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None)
    model_options = []
    for key in ("regression", "neural"):
        label = MODEL_NAMES[key] + ("" if key in AVAILABLE_MODELS else " (only available in the local version)")
        model_options.append({"label": label, "value": key, "disabled": key not in AVAILABLE_MODELS})

    return html.Div(
        [
            dcc.Store(id="metric-store"),  # lives in this tab, so every visit starts without old results
            explainer(),
            card(
                "Forecast against measured power",
                "Test year 2019, one prediction per hour. Pick a model and the inputs, then run the forecast.",
                html.Div(
                    [
                        control("Forecasting model", choice_group("model-selection-dropdown", model_options, default_model)),
                        control("Inputs of the forecast", choice_group("forecast-inputs", [{"label": f, "value": f} for f in features], features, multi=True)),
                        html.Button("Run forecast", id="run-prediction-button", n_clicks=0, className="primary"),
                    ],
                    className="controls",
                ),
                html.P(
                    "All four inputs use the saved model. With fewer inputs a simple Random Forest is trained on 2017–2018 with only those inputs, "
                    "because the saved models need all four.",
                    className="hint-line",
                ),
                html.Div(notice("Click “Run forecast” to compute the prediction for 2019."), id="custom-prediction-output"),
                html.Hr(),
                control(
                    "Metrics to display",
                    choice_group(
                        "metric-choice",
                        [{"label": f"{k} · {v[0]}", "value": k} for k, v in METRIC_INFO.items()],
                        ["MAE", "RMSE", "cvRMSE", "NMBE"],
                        multi=True,
                    ),
                ),
                html.Div(id="custom-metrics-output"),
            ),
        ]
    )


def raw_tab():
    first, last = df_2019["Date"].min(), df_2019["Date"].max()
    return card(
        "Raw data 2019",
        f"Measured weather and power of the test year, {first:%d %b} to {last:%d %b %Y}.",
        data_section("raw", "raw-weather-variable"),
    )


def features_tab():
    return card(
        "Feature selection",
        "Which inputs say the most about the power? Choose the features and the method (scores for the 2019 power).",
        html.Div(
            [
                control("Features", choice_group("feature-select-dropdown", [{"label": f, "value": f} for f in features], features[:2], multi=True)),
                control(
                    "Method",
                    choice_group(
                        "feature-method-dropdown",
                        [
                            {"label": "F-test (f_regression)", "value": "f_regression"},
                            {"label": "Mutual information", "value": "mutual_info"},
                        ],
                        "f_regression",
                    ),
                ),
                html.Button("Run feature selection", id="feature-button", n_clicks=0, className="primary"),
            ],
            className="controls",
        ),
        html.Div(id="feature-importance-graph"),
    )


def training_tab():
    return card(
        "Training data 2017–2018",
        "The data the models learned from. Shown for reference; the forecast itself is on the first tab.",
        control(
            "Year",
            choice_group("training-year", [{"label": "2017", "value": 2017}, {"label": "2018", "value": 2018}], 2017),
        ),
        data_section("training", "training-weather-variable"),
    )


@app.callback(Output("content-container", "children"), Input("main-tab", "value"))
def render_tab(tab):
    return {"forecast": forecast_tab, "raw": raw_tab, "features": features_tab, "training": training_tab}.get(tab, forecast_tab)()


# --------------------------------------------------------------------------------------
# Raw data graphs (2019 tab and training tab share the same drawing code)
# --------------------------------------------------------------------------------------
def build_data_graphs(year, weather_var):
    if year not in DATA_BY_YEAR:
        return html.Div(), html.Div(), html.Div()
    if weather_var not in features[1:]:
        weather_var = features[1]
    df = DATA_BY_YEAR[year]

    weather_fig = px.line(df, x="Date", y=weather_var)
    weather_fig.update_traces(line=dict(width=1.2, color=BLUE))
    style_figure(weather_fig, f"{weather_var} · {year}", weather_var, str(year))
    month_axis(weather_fig)

    last_hour = html.Div()
    if year == 2019:
        last_fig = px.line(df, x="Date", y="Last Hour Power in kW")
        last_fig.update_traces(line=dict(width=1.2, color=GREY))
        style_figure(last_fig, "Last hour power in kW · 2019", "Power in kW", "2019")
        month_axis(last_fig)
        last_hour = graph(last_fig)

    power_fig = px.line(df, x="Date", y="Power in kW")
    power_fig.update_traces(line=dict(width=1.2, color=ORANGE))
    style_figure(power_fig, f"Power consumption in kW · {year}" + (" (measured)" if year == 2019 else ""), "Power in kW", str(year))
    month_axis(power_fig)

    return graph(weather_fig), last_hour, graph(power_fig)


def safe_data_graphs(year, weather_var):
    try:
        return build_data_graphs(year, weather_var)
    except Exception as exc:  # show the cause on the page instead of a bare "Callback error"
        traceback.print_exc()
        return notice(f"The graphs could not be drawn: {type(exc).__name__}: {exc}", "error"), html.Div(), html.Div()


@app.callback(
    Output("raw-weather-graph", "children"),
    Output("raw-last-hour-graph", "children"),
    Output("raw-power-graph", "children"),
    Input("raw-weather-variable", "value"),
)
def update_raw_graphs(weather_var):
    return safe_data_graphs(2019, weather_var)


@app.callback(
    Output("training-weather-graph", "children"),
    Output("training-last-hour-graph", "children"),
    Output("training-power-graph", "children"),
    Input("training-year", "value"),
    Input("training-weather-variable", "value"),
)
def update_training_graphs(year, weather_var):
    return safe_data_graphs(year, weather_var)

# --------------------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------------------
def _mutual_info(X, y):
    return mutual_info_regression(X, y, random_state=0)


@app.callback(
    Output("feature-importance-graph", "children"),
    Input("feature-button", "n_clicks"),
    State("feature-select-dropdown", "value"),
    State("feature-method-dropdown", "value"),
    prevent_initial_call=True,
)
def run_feature_selection(n_clicks, selected_features, method):
    if not selected_features:
        return notice("Please select at least one feature.", "error")

    X = df_2019[selected_features]
    y = df_2019["Power in kW"]

    selector = SelectKBest(score_func=f_regression if method == "f_regression" else _mutual_info, k="all")
    selector.fit(X, y)
    order = np.argsort(selector.scores_)[::-1]
    names = [selected_features[i] for i in order]
    scores = [float(selector.scores_[i]) for i in order]
    label = "F-score" if method == "f_regression" else "Mutual information score"

    fig = go.Figure(go.Bar(x=names, y=scores, marker_color=BLUE, text=[f"{s:,.2f}" for s in scores], textposition="outside"))
    style_figure(fig, "Feature importance · " + ("F-test" if method == "f_regression" else "mutual information"), label, height=420)
    fig.update_layout(hovermode="closest", showlegend=False, margin=dict(l=64, r=24, t=64, b=60))
    fig.update_yaxes(rangemode="tozero")
    return graph(fig)


# --------------------------------------------------------------------------------------
# Forecast and metrics
# --------------------------------------------------------------------------------------
@app.callback(
    Output("custom-prediction-output", "children"),
    Output("metric-store", "data"),
    Output("run-prediction-button", "children"),
    Input("run-prediction-button", "n_clicks"),
    State("model-selection-dropdown", "value"),
    State("forecast-inputs", "value"),
    prevent_initial_call=True,
)
def run_model_prediction(n_clicks, model_choice, selected_inputs):
    used = [f for f in features if f in (selected_inputs or [])]  # keeps the order the models were trained with
    if not used:
        return notice("Select at least one input for the forecast.", "error"), None, no_update
    if len(used) == len(features) and model_choice not in AVAILABLE_MODELS:
        return notice("No model file found next to the script.", "error"), None, no_update

    try:
        output, stored = compute_forecast(model_choice, used)
        return output, stored, "Update forecast"  # from now on the button updates an existing forecast
    except Exception as exc:  # show the cause on the page instead of a bare "Callback error"
        traceback.print_exc()
        versions = f"numpy {np.__version__}, scikit-learn {sklearn.__version__}"
        return notice(f"The forecast could not be computed: {type(exc).__name__}: {exc} ({versions})", "error"), None, no_update


def compute_forecast(model_choice, used):
    if len(used) == len(features):
        model, label = load_model(model_choice), MODEL_NAMES[model_choice]
        setup = f"Inputs: {', '.join(used)}. Model: {label} (saved model)."
    else:
        model, label = train_quick_model(tuple(used)), f"Random Forest (retrained, {len(used)} of {len(features)} inputs)"
        setup = f"Inputs: {', '.join(used)}. Model: simple Random Forest trained on 2017–2018 with only these inputs, not the saved model."

    df_temp = df_2019.copy()
    df_temp["Model Prediction"] = np.asarray(model.predict(df_temp[used])).ravel()

    fig = px.line(df_temp, x="Date", y=["Power in kW", "Model Prediction"])
    fig.data[0].update(name="Measured power", line=dict(width=1.2, color=BLUE))
    fig.data[1].update(name="Model prediction", line=dict(width=1.2, color=ORANGE))
    style_figure(fig, f"{label} vs measured power · 2019", "Power in kW", "Date", height=480)

    actual, predicted = df_temp["Power in kW"], df_temp["Model Prediction"]
    mbe = float(np.mean(actual - predicted))
    mse = float(metrics.mean_squared_error(actual, predicted))
    rmse = float(np.sqrt(mse))
    mean_power = float(np.mean(actual))
    values = {
        "MAE": float(metrics.mean_absolute_error(actual, predicted)),
        "MBE": mbe,
        "MSE": mse,
        "RMSE": rmse,
        "cvRMSE": rmse / mean_power,
        "NMBE": mbe / mean_power,
    }
    return html.Div([notice(setup), graph(fig)]), {"model": label, "values": values}


@app.callback(
    Output("custom-metrics-output", "children"),
    Input("metric-choice", "value"),
    Input("metric-store", "data"),
)
def display_selected_metrics(selected_keys, data):
    if not data:
        return notice("Run the forecast to see the metrics.")
    if not selected_keys:
        return notice("Select at least one metric.")

    tiles = []
    for key in METRIC_INFO:  # keeps a fixed order
        if key not in selected_keys:
            continue
        name, unit, factor = METRIC_INFO[key]
        hint = METRIC_HINT[key]
        tiles.append(
            html.Div(
                [
                    html.Div(key, className="metric-key"),
                    html.Div([f"{data['values'][key] * factor:,.2f}", html.Span(f" {unit}", className="metric-unit")], className="metric-value"),
                    html.Div(name, className="metric-name"),
                    html.Div(hint, className="metric-hint"),
                ],
                className="metric-tile",
            )
        )
    return html.Div(
        [html.Div(f"Metrics of the {data['model']} model on the 2019 data", className="metric-heading"), html.Div(tiles, className="metric-grid")]
    )


if __name__ == "__main__":
    app.run(debug=False)
