"""
Central configuration for the NSE/BSE Options Trading Bot.
All secrets come from environment variables — never hardcode credentials.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Broker credentials
# ---------------------------------------------------------------------------

@dataclass
class ZerodhaConfig:
    api_key: str = os.getenv("ZERODHA_API_KEY", "")
    api_secret: str = os.getenv("ZERODHA_API_SECRET", "")
    access_token: str = os.getenv("ZERODHA_ACCESS_TOKEN", "")
    user_id: str = os.getenv("ZERODHA_USER_ID", "")


@dataclass
class FyersConfig:
    client_id: str = os.getenv("FYERS_CLIENT_ID", "")
    secret_key: str = os.getenv("FYERS_SECRET_KEY", "")
    access_token: str = os.getenv("FYERS_ACCESS_TOKEN", "")
    redirect_uri: str = os.getenv("FYERS_REDIRECT_URI", "http://127.0.0.1:5000/")


@dataclass
class AngelOneConfig:
    api_key: str = os.getenv("ANGEL_API_KEY", "")
    client_id: str = os.getenv("ANGEL_CLIENT_ID", "")
    password: str = os.getenv("ANGEL_PASSWORD", "")
    totp_key: str = os.getenv("ANGEL_TOTP_KEY", "")


# ---------------------------------------------------------------------------
# Market settings
# ---------------------------------------------------------------------------

@dataclass
class MarketConfig:
    # Exchanges
    primary_exchange: str = "NSE"
    secondary_exchange: str = "BSE"

    # Indices to trade options on
    indices: List[str] = field(default_factory=lambda: [
        "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"
    ])

    # Underlying stocks (F&O eligible)
    top_stocks: List[str] = field(default_factory=lambda: [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "KOTAKBANK", "AXISBANK", "SBIN", "BAJFINANCE", "BAJAJFINSV",
        "LT", "ITC", "HINDUNILVR", "ASIANPAINT", "MARUTI",
        "TITAN", "WIPRO", "HCLTECH", "TECHM", "SUNPHARMA",
    ])

    # Market hours IST
    market_open: str = "09:15"
    market_close: str = "15:30"
    pre_market_start: str = "09:00"
    post_market_end: str = "16:00"

    # Lot sizes (update as SEBI revises)
    lot_sizes: dict = field(default_factory=lambda: {
        "NIFTY": 50, "BANKNIFTY": 15, "FINNIFTY": 40,
        "MIDCPNIFTY": 75, "SENSEX": 10,
    })

    # Expiry preference
    preferred_expiry: str = "weekly"   # 'weekly' | 'monthly'
    expiry_day: str = "thursday"       # NSE weekly expiry


# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    active_strategies: List[str] = field(default_factory=lambda: [
        "iron_condor",
        "short_straddle",
        "short_strangle",
        "bull_put_spread",
        "bear_call_spread",
        "momentum_directional",
        "vega_scalping",
        "delta_neutral",
    ])

    # Position sizing
    max_capital_per_trade: float = 0.05   # 5% of portfolio per trade
    max_open_positions: int = 10
    max_delta_exposure: float = 50        # total portfolio delta
    max_vega_exposure: float = 5000

    # Greeks thresholds for entry
    target_iv_rank_min: float = 30.0      # Enter short vol when IVR > 30
    target_iv_rank_max: float = 100.0
    min_dte: int = 3                      # Minimum days to expiry
    max_dte: int = 45                     # Maximum days to expiry
    preferred_dte: int = 21

    # Exit rules
    profit_target_pct: float = 0.50       # Close at 50% profit
    stop_loss_pct: float = 2.00           # Close at 2x premium received
    dte_exit: int = 2                     # Exit 2 DTE regardless

    # Delta ranges for strikes
    short_delta_range: tuple = (0.15, 0.35)
    long_delta_range: tuple = (0.05, 0.15)


# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    max_portfolio_loss_daily: float = 0.03      # 3% daily drawdown limit
    max_portfolio_loss_weekly: float = 0.07     # 7% weekly drawdown limit
    max_portfolio_loss_monthly: float = 0.15    # 15% monthly drawdown limit
    max_single_position_loss: float = 0.02      # 2% per position
    circuit_breaker_pct: float = 0.05           # Halt trading at 5% daily loss
    position_correlation_limit: float = 0.70    # Max correlation between positions
    min_liquidity_oi: int = 500                 # Min open interest
    min_liquidity_volume: int = 100             # Min daily volume
    max_bid_ask_spread_pct: float = 0.02        # Max 2% bid-ask spread


# ---------------------------------------------------------------------------
# ML / Research config
# ---------------------------------------------------------------------------

@dataclass
class ResearchConfig:
    # Auto-research schedule
    research_interval_hours: int = 4            # Run research every 4h
    backtest_lookback_days: int = 365           # 1 year backtest window
    walk_forward_periods: int = 12             # 12-period walk-forward
    min_trades_for_significance: int = 30       # Min trades for stats

    # Feature engineering
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    iv_percentile_window: int = 252             # 1 year for IVR calc

    # Model registry
    model_dir: str = "models/"
    retrain_frequency_days: int = 7            # Retrain weekly

    # Optuna hyperparameter tuning
    optuna_trials: int = 100
    optuna_timeout_seconds: int = 3600

    # Performance thresholds to deploy new model
    min_sharpe_ratio: float = 1.5
    min_win_rate: float = 0.55
    max_max_drawdown: float = 0.20


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@dataclass
class DatabaseConfig:
    postgres_url: str = os.getenv(
        "DATABASE_URL", "postgresql://trader:trader@localhost:5432/options_bot"
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    mongo_url: str = os.getenv("MONGO_URL", "mongodb://localhost:27017/options_research")


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

@dataclass
class NotificationConfig:
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enable_telegram: bool = bool(os.getenv("TELEGRAM_BOT_TOKEN"))
    alert_on_trade: bool = True
    alert_on_daily_summary: bool = True
    alert_on_circuit_breaker: bool = True


# ---------------------------------------------------------------------------
# Composite config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    zerodha: ZerodhaConfig = field(default_factory=ZerodhaConfig)
    fyers: FyersConfig = field(default_factory=FyersConfig)
    angel: AngelOneConfig = field(default_factory=AngelOneConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    # Active broker
    active_broker: str = os.getenv("ACTIVE_BROKER", "zerodha")  # zerodha|fyers|angel
    paper_trading: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    timezone: str = "Asia/Kolkata"


# Singleton
config = Config()
