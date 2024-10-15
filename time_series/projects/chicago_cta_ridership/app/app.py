import os
import joblib
from io import StringIO

import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd

from src.opt import optimize_vehicle_allocation
from src.base_trainer import BaseTrainer

# Load forecast results
all_forecast_results = joblib.load(os.path.join("app/data", "forecast_results.joblib"))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

# Paramaters for optimization
fixed_opt_params = {
    "cost_bus_40": 150,
    "cost_bus_60": 180,
    "cost_rail": 300,
    "capacity_bus_40": 53,
    "capacity_bus_60": 80,
    "capacity_rail": 80,
    "trips_per_bus": 5,
    "trips_per_rail": 3,
    "percentage_40_foot": 0.8,
}

# ---------------------------- Application layout ---------------------------- #

app.layout = dbc.Container(
    [
        # Explanations and inputs in the same row
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P("Select a forecasting model:"),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[
                                {"label": "ETS", "value": "ets"},
                                {"label": "SARIMAX", "value": "sarimax"},
                                {"label": "VAR", "value": "var"},
                            ],
                            value="ets",
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        # Plot output
        dbc.Row(
            dbc.Col(dcc.Graph(id="forecast-plot"), width={"size": 10}), justify="center"
        ),
        # Download button at the top
        dbc.Row(
            dbc.Col(
                html.Button("Download CSV", id="download-button"), width={"size": 2}
            ),
            justify="left-justify",
        ),
        # Allocation table output
        dbc.Row(
            dbc.Col(
                dash_table.DataTable(
                    id="results-table",
                    style_cell={"textAlign": "center"},
                    style_header={"backgroundColor": "white", "fontWeight": "bold"},
                ),
                width=12,
            ),
            justify="center",
        ),
        dcc.Download(id="download-csv"),
        dcc.Store(id="stored-results"),
    ],
    fluid=True,
)

# ---------------------------- Callbacks ---------------------------- #


@app.callback(
    [
        Output("forecast-plot", "figure"),
        Output("results-table", "data"),
        Output("stored-results", "data"),
    ],
    [Input("model-dropdown", "value")],
)
def update_output(selected_model):
    # Run the forecast
    forecast_results = all_forecast_results[f"forecast_results_{selected_model}"]

    # Plot the forecast
    plot = BaseTrainer.plot_forecast(
        start_date=forecast_results["y_train"].index.max() - pd.DateOffset(days=30),
        pi=forecast_results["pi"],
        y_train=forecast_results["y_train"],
        y_pred=forecast_results["y_pred"],
        static=False,
        title=f"CTA Forecast - Short Term (Out-of-Sample Forecast) - {selected_model.upper()}",
    )

    forecast = forecast_results["y_pred"]

    # Optimization function call
    opt_results = optimize_vehicle_allocation(forecast=forecast, **fixed_opt_params)

    # Create a data table for the results
    results_data = opt_results.to_dict("records")

    return plot, results_data, opt_results.to_json(date_format="iso", orient="split")


@app.callback(
    Output("download-csv", "data"),
    Input("download-button", "n_clicks"),
    State("stored-results", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, stored_data):
    if stored_data is None:
        raise dash.exceptions.PreventUpdate
    results = pd.read_json(StringIO(stored_data), orient="split")
    return dcc.send_data_frame(results.to_csv, "allocations.csv")


if __name__ == "__main__":
    app.run_server(debug=True)
