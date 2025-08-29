# trading_system/main.py
"""Main entry point for the trading system.

This module configures logging, loads tickers, and starts the data collector.
"""

import logging
import sys

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from src.collectors import data_collector
    from utils.file_utils import load_tickers_from_yaml

    tickers = load_tickers_from_yaml("data/tickers.yml")
    data_collector.main(tickers)
    print(tickers)
except FileNotFoundError as e:
    sys.exit(f"File not found: {e}")
except ImportError as e:
    sys.exit(f"Import error: {e}")
