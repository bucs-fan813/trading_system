"""
Main script to run hyperparameter optimization and sensitivity analysis
for the Know Sure Thing strategy using the portfolio-based evaluation framework.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

print(os.getcwd())

sys.path.insert(0, os.path.abspath(os.getcwd()))

try:
    from src.database.config import DatabaseConfig
    from src.optimizer.sensitivity_analyzer import SensitivityAnalyzer
    from src.optimizer.strategy_optimizer import StrategyOptimizer
    from src.strategies.advanced.garchx_strat import GarchXStrategyStrategy
except ImportError as e:
    print("Error importing modules. Make sure the script is run from the project root")
    print("or the 'src' directory is in the Python path.")
    print(f"Import Error: {e}")
    sys.exit(1)

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Data Configuration
TICKER_FILE_PATH = "data/ticker.xlsx" # Path relative to project root
MAX_TICKERS = 1 # Limit tickers for faster testing, set to None to use all

# Backtest Period
START_DATE = (datetime.now() - timedelta(days=4*365)).strftime("%Y-%m-%d")
END_DATE = datetime.now().strftime("%Y-%m-%d")
# --- Helper Functions ---

def load_tickers(file_path: str, max_tickers: Optional[int] = None) -> List[str]:
    """Loads and formats ticker symbols from an Excel file."""
    tickers_df = pd.read_excel(file_path)
    # Basic validation
    if not all(col in tickers_df.columns for col in ["Security Name"]):
        raise ValueError("Ticker file missing required columns: 'Security Name'")

    tickers_df = tickers_df.drop_duplicates(subset=["Security Name"]).reset_index(drop=True)

    def add_ticker_suffix(row):
        name = str(row["Security Name"]).strip().upper()
        # Fetch company information using yfinance
        stock = yf.Ticker(name)
        exchange = str(stock.info.get("exchange", None)).strip().upper()
        return f"{name}"

    tickers_df["Ticker"] = tickers_df.apply(add_ticker_suffix, axis=1)
    ticker_list = tickers_df["Ticker"].unique().tolist()

    if max_tickers and len(ticker_list) > max_tickers:
        ticker_list = ticker_list[:max_tickers]

    if not ticker_list:
        raise ValueError("No tickers loaded.")

    return ticker_list

tickers_to_run = load_tickers(TICKER_FILE_PATH, MAX_TICKERS)

tickers_to_run = tickers_to_run[:10]

db_config = DatabaseConfig.default()

cc = GarchXStrategyStrategy(db_config=db_config)

b = cc.generate_signals(ticker=tickers_to_run, start_date=START_DATE, end_date=END_DATE)
