from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Load data
df_2017_2018 = pd.read_csv("regression_data_2017-2018.csv")
df_2019 = pd.read_csv("prepared_test_data_2019.csv")
# model_path_default = "not used in online version"
model_path_nn = "neural_networks_model.pkl"

# Preprocess
def preprocess_df(df):
    df.rename(columns={"Solar Radiation in W/m^2": "Solar Radiation in W/m²"}, inplace=True)
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

df_2017_2018 = preprocess_df(df_2017_2018)
df_2019 = preprocess_df(df_2019)

df_2017 = df_2017_2018[df_2017_2018['Date'].dt.year == 2017]
df_2018 = df_2017_2018[df_2017_2018['Date'].dt.year == 2018]

features = ["Last Hour Power in kW", "Temperature in °C", "Humidity in %", "Solar Radiation in W/m²"]

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Energy Dashboard"

weather_options = [{'label': f, 'value': f} for f in features[1:]]

metric_labels = {
    "MAE": "Mean Absolute Error (MAE)",
    "MBE": "Mean Bias Error (MBE)",
    "MSE": "Mean Squared Error (MSE)",
    "RMSE": "Root Mean Squared Error (RMSE)",
    "cvRMSE": "Coefficient of Variation RMSE (cvRMSE)",
    "NMBE": "Normalized Mean Bias Error (NMBE)"
}

app.layout = html.Div([
    html.Div(className="hero-container", children=[
        html.H1("Welcome to my Energy Dashboard for the Central IST Building"),
        html.P("Please select the year you would like to observe:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(y), 'value': y} for y in [2017, 2018, 2019]],
            value=2017,
            clearable=False,
            className='dash-dropdown'
        ),
        html.Div(id='year-section-title')
    ]),
    html.Div(id='content-container')
])

@app.callback(
    Output('year-section-title', 'children'),
    Input('year-dropdown', 'value')
)
def update_year_section_title(year):
    return html.H2("Model Training Data") if year in [2017, 2018] else html.H2("Prediction of the Power Consumption for 2019")

@app.callback(
    Output('content-container', 'children'),
    Input('year-dropdown', 'value')
)
def render_content_by_year(year):
    dropdown = html.Div([
        html.Label("Select the weather variable you would like to observe:"),
        dcc.Dropdown(
            id='weather-variable-dropdown',
            options=weather_options,
            value=weather_options[0]['value'],
            clearable=False
        )
    ], className='card-container')

    if year in [2017, 2018]:
        return html.Div([
            dropdown,
            html.Div(id='weather-graph-container'),
            html.Div(id='last-hour-graph-container'),
            html.Div(id='power-graph-container'),
        ])
    else:
        return html.Div([
            html.Div("Raw Data 2019", className="section-header"),
            dropdown,
            html.Div(id='weather-graph-container'),
            html.Div(id='last-hour-graph-container'),

            html.Hr(),
            html.Div("Feature Selection", className="section-header"),
            html.Div([
                html.Div([
                    html.Label("Select the features you would like to include in the Feature Selection:"),
                    dcc.Dropdown(
                        id='feature-select-dropdown',
                        options=[{'label': f, 'value': f} for f in features],
                        value=features[:2],
                        multi=True
                    )
                ]),
                html.Div([
                    html.Label("Select the feature Selection method you would like to use:"),
                    dcc.Dropdown(
                        id='feature-method-dropdown',
                        options=[
                            {'label': 'F-test (f_regression)', 'value': 'f_regression'},
                            {'label': 'Mutual Information Regression', 'value': 'mutual_info'}
                        ],
                        value='f_regression'
                    )
                ]),
                html.Button("Run Feature Selection", id="feature-button", n_clicks=0)
            ], className='card-container'),

            html.Div(id='feature-importance-graph'),

            html.Hr(),
            html.Div("Model Based Prediction", className="section-header"),
            html.Div([
                html.Label("Select the model that should be used to make the prediction:"),
                dcc.Dropdown(
                    id='model-selection-dropdown',
                    options=[
                        {'label': 'Random Forest Model (Not available online)', 'value': 'regression', 'disabled': True},
                        {'label': 'Neural Network Model', 'value': 'neural'}
                    ],
                    placeholder="Select a model...",
                    style={'marginBottom': '10px'}
                ),
                html.Button("Run Prediction", id="run-prediction-button", n_clicks=0)
            ], className='card-container'),

            html.Div(id='custom-prediction-output'),
            html.Div(id='metric-selection-ui'),
            html.Div(id='custom-metrics-output')
        ])

@app.callback(
    Output('weather-graph-container', 'children'),
    Output('last-hour-graph-container', 'children'),
    Output('power-graph-container', 'children'),
    Input('year-dropdown', 'value'),
    Input('weather-variable-dropdown', 'value')
)
def update_graphs(year, weather_var):
    df = df_2017 if year == 2017 else df_2018 if year == 2018 else df_2019
    month_ticks = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='MS')

    weather_fig = px.line(df, x='Date', y=weather_var, height=675, width=2000)
    weather_fig.update_layout(title=f"{weather_var} - {year}",
                              title_x=0.5,
                              xaxis=dict(tickmode='array', tickvals=month_ticks, tickformat="%b"),
                              yaxis_title=weather_var,
                              xaxis_title=year)

    last_hour = html.Div()
    if year == 2019:
        last_hour_fig = px.line(df, x='Date', y="Last Hour Power in kW", height=675, width=2000)
        last_hour_fig.update_layout(
            title="Last Hour Power in kW - 2019",
            title_x=0.5,
            xaxis=dict(tickmode='array', tickvals=month_ticks, tickformat="%b"),
            yaxis_title="Power in kW",
            xaxis_title=year
        )
        last_hour = html.Div(dcc.Graph(figure=last_hour_fig), style={'display': 'flex', 'justifyContent': 'center'})

    power_graph = html.Div()
    if year in [2017, 2018]:
        power_fig = px.line(df, x='Date', y='Power in kW', height=675, width=2000)
        power_fig.update_layout(
            title=f"Power Consumption in kW - {year}",
            title_x=0.5,
            xaxis=dict(tickmode='array', tickvals=month_ticks, tickformat="%b"),
            yaxis_title="Power in kW",
            xaxis_title=year
        )
        power_graph = html.Div(dcc.Graph(figure=power_fig), style={'display': 'flex', 'justifyContent': 'center'})

    return html.Div(dcc.Graph(figure=weather_fig), style={'display': 'flex', 'justifyContent': 'center'}), last_hour, power_graph

@app.callback(
    Output('feature-importance-graph', 'children'),
    Input('feature-button', 'n_clicks'),
    State('feature-select-dropdown', 'value'),
    State('feature-method-dropdown', 'value'),
    prevent_initial_call=True
)
def run_feature_selection(n_clicks, selected_features, method):
    if not selected_features:
        return html.Div("Please select at least 1 feature.", style={'textAlign': 'center', 'color': 'red'})

    X = df_2019[selected_features]
    y = df_2019["Power in kW"]

    selector = SelectKBest(score_func=f_regression if method == "f_regression" else mutual_info_regression, k='all')
    selector.fit(X, y)
    scores = selector.scores_
    ylabel = "F-score" if method == "f_regression" else "MI Score"

    fig = go.Figure(go.Bar(x=selected_features, y=scores))
    fig.update_layout(title=f"Feature Importance using {method}",
                      title_x=0.5,
                      xaxis_title="Feature", yaxis_title=ylabel,
                      xaxis_tickangle=45, height=500)

    return html.Div(dcc.Graph(figure=fig), style={'width': '70%', 'margin': '0 auto'})

@app.callback(
    Output('custom-prediction-output', 'children'),
    Output('metric-selection-ui', 'children'),
    Input('run-prediction-button', 'n_clicks'),
    State('model-selection-dropdown', 'value'),
    prevent_initial_call=True
)
def run_model_prediction(n_clicks, model_choice):
    if model_choice != 'neural':
        return html.Div("Please select a valid model.", style={'textAlign': 'center', 'color': 'red'}), html.Div()

    model = joblib.load(model_path_nn)

    df_temp = df_2019.copy()
    df_temp["Model Prediction"] = model.predict(df_temp[features])

    fig = px.line(df_temp, x='Date', y=["Power in kW", "Model Prediction"],
                  title="Neural Network Model vs Actual")
    fig.update_layout(height=600, title_x=0.5, xaxis_title="Date", yaxis_title="Power in kW",
                      legend=dict(orientation="h", y=-0.3, x=0.5, xanchor='center'))

    mae = metrics.mean_absolute_error(df_temp["Power in kW"], df_temp["Model Prediction"])
    mbe = np.mean(df_temp["Power in kW"] - df_temp["Model Prediction"])
    mse = metrics.mean_squared_error(df_temp["Power in kW"], df_temp["Model Prediction"])
    rmse = np.sqrt(mse)
    cvrmse = rmse / np.mean(df_temp["Power in kW"])
    nmbe = mbe / np.mean(df_temp["Power in kW"])

    app.server.metric_values = {
        "MAE": mae,
        "MBE": mbe,
        "MSE": mse,
        "RMSE": rmse,
        "cvRMSE": cvrmse,
        "NMBE": nmbe
    }

    checklist = html.Div([
        html.Label("Select metrics to display:"),
        dcc.Checklist(
            id='metric-choice',
            options=[{'label': metric_labels[k], 'value': k} for k in metric_labels],
            value=['MAE'],
            labelStyle={'display': 'block'}
        )
    ], className='card-container', style={'width': '350px', 'margin': '20px auto'})

    return html.Div(dcc.Graph(figure=fig), style={'width': '80%', 'margin': '20px auto'}), checklist

@app.callback(
    Output('custom-metrics-output', 'children'),
    Input('metric-choice', 'value'),
    prevent_initial_call=True
)
def display_selected_metrics(selected_keys):
    if not selected_keys:
        return html.Div("No metrics selected.", className='card-container', style={'textAlign': 'center'})

    table_rows = [
        html.Tr([html.Th("Metric"), html.Th("Value")])
    ] + [
        html.Tr([html.Td(k), html.Td(f"{app.server.metric_values[k]:.4f}")]) for k in selected_keys
    ]

    return html.Div([
        html.Table(table_rows, style={'width': '100%', 'borderCollapse': 'collapse', 'textAlign': 'center'})
    ], className='card-container', style={'width': '400px', 'margin': '20px auto', 'fontSize': '16px'})

# Required for Render
server = app.server
#if __name__ == '__main__':

  #  app.run(debug=True)
