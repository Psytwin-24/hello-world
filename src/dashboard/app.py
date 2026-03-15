"""
Real-time trading dashboard — Plotly Dash web app.

Features:
  - Live P&L curve
  - Open positions with live Greeks
  - Option chain heatmap
  - Strategy performance breakdown
  - Risk metrics panel
  - Research cycle history
  - Trade log

Run with: python -m src.dashboard.app
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import dash
    from dash import Input, Output, callback, dcc, html
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


def build_layout():
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col(html.H2("NSE/BSE Options Bot", className="text-white"), width=8),
            dbc.Col(html.Div(id="market-status", className="text-right text-white"), width=4),
        ], className="bg-dark p-3 mb-3"),

        # KPI Row
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Daily P&L", className="text-muted"),
                    html.H4(id="daily-pnl", className="text-success"),
                ])
            ]), width=2),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Total P&L", className="text-muted"),
                    html.H4(id="total-pnl"),
                ])
            ]), width=2),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Open Positions", className="text-muted"),
                    html.H4(id="open-positions"),
                ])
            ]), width=2),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Win Rate (30d)", className="text-muted"),
                    html.H4(id="win-rate"),
                ])
            ]), width=2),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("India VIX", className="text-muted"),
                    html.H4(id="india-vix"),
                ])
            ]), width=2),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Circuit Breaker", className="text-muted"),
                    html.H4(id="circuit-breaker"),
                ])
            ]), width=2),
        ], className="mb-3"),

        # Charts Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Equity Curve"),
                    dbc.CardBody(dcc.Graph(id="equity-curve", style={"height": "300px"})),
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Strategy Allocation"),
                    dbc.CardBody(dcc.Graph(id="strategy-pie", style={"height": "300px"})),
                ])
            ], width=4),
        ], className="mb-3"),

        # Option Chain + Greeks
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Option Chain — ",
                        dcc.Dropdown(
                            id="symbol-selector",
                            options=[
                                {"label": s, "value": s}
                                for s in ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]
                            ],
                            value="NIFTY",
                            style={"width": "150px", "display": "inline-block"},
                        ),
                    ]),
                    dbc.CardBody(dcc.Graph(id="option-chain-heatmap", style={"height": "400px"})),
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Portfolio Greeks"),
                    dbc.CardBody(html.Div(id="portfolio-greeks")),
                ])
            ], width=4),
        ], className="mb-3"),

        # Positions Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Open Positions"),
                    dbc.CardBody(html.Div(id="positions-table")),
                ])
            ], width=12),
        ], className="mb-3"),

        # Research + Trade Log
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Research Cycle Log"),
                    dbc.CardBody(html.Div(id="research-log")),
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent Trades"),
                    dbc.CardBody(html.Div(id="trade-log")),
                ])
            ], width=6),
        ]),

        # Auto-refresh
        dcc.Interval(id="refresh", interval=30_000, n_intervals=0),  # 30s
        dcc.Store(id="state-store"),

    ], fluid=True, className="bg-light")


def create_equity_chart(equity_data: List[Dict]) -> go.Figure:
    if not equity_data:
        return go.Figure()

    df = pd.DataFrame(equity_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["portfolio"],
        mode="lines", name="Portfolio",
        line=dict(color="#00bc8c", width=2),
        fill="tonexty",
    ))
    fig.add_hline(
        y=df["portfolio"].iloc[0] if len(df) > 0 else 0,
        line_dash="dash", line_color="gray",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def create_chain_heatmap(chain_df: pd.DataFrame, spot: float) -> go.Figure:
    if chain_df is None or chain_df.empty:
        return go.Figure()

    nearest_expiry = chain_df["expiry"].iloc[0] if "expiry" in chain_df.columns else ""
    df = chain_df[chain_df["expiry"] == nearest_expiry].copy()

    # Focus on strikes within 5% of spot
    df = df[
        (df["strike"] >= spot * 0.95) & (df["strike"] <= spot * 1.05)
    ].sort_values("strike")

    calls = df[df["type"] == "CE"].set_index("strike")
    puts = df[df["type"] == "PE"].set_index("strike")

    strikes = sorted(set(calls.index) | set(puts.index))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Calls (CE)", "Puts (PE)"],
                        shared_yaxes=True)

    if not calls.empty:
        fig.add_trace(go.Bar(
            x=calls.reindex(strikes)["oi"].fillna(0),
            y=[str(int(s)) for s in strikes],
            orientation="h", name="Call OI",
            marker_color=["red" if s <= spot else "lightblue" for s in strikes],
        ), row=1, col=1)

    if not puts.empty:
        fig.add_trace(go.Bar(
            x=puts.reindex(strikes)["oi"].fillna(0),
            y=[str(int(s)) for s in strikes],
            orientation="h", name="Put OI",
            marker_color=["lightgreen" if s >= spot else "orange" for s in strikes],
        ), row=1, col=2)

    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=30, b=0),
        barmode="overlay",
    )
    return fig


def launch_dashboard(port: int = 8050, debug: bool = False):
    """Start the Dash server."""
    if not DASH_AVAILABLE:
        logger.error("Dash not installed. Run: pip install dash dash-bootstrap-components")
        return

    from loguru import logger
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout()

    # Callbacks would connect to the live bot state here
    # (injected via shared Redis state or in-memory state store)

    logger.info(f"Dashboard running at http://localhost:{port}")
    app.run(debug=debug, port=port, host="0.0.0.0")


if __name__ == "__main__":
    launch_dashboard(debug=True)
