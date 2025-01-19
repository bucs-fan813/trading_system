# trading_system/collectors/info_collector.py

import logging
import yfinance as yf
import pandas as pd
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict
from src.collectors.base_collector import BaseCollector
# from concurrent.futures import ThreadPoolExecutor
# TODO use threading to speed up the process

logger = logging.getLogger(__name__)

class InfoCollector:
    """Collect company information using yfinance and store it in the database."""

    def _flatten_nested_dict(self, nested_dict: Dict) -> Dict:
        """Flatten a nested dictionary structure."""
        flat_dict = {}
        def flatten(d, parent_key=''):
            for k, v in d.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten(v, new_key)
                else:
                    flat_dict[new_key] = v
        flatten(nested_dict)
        return flat_dict

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_company_info(self, ticker: str) -> None:
        """Fetch and store company information."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                logger.warning(f"No company info available for {ticker}")
                return

            # Flatten nested info structure
            flat_info = self._flatten_nested_dict(info)
            
            # Convert to DataFrame
            info_df = pd.DataFrame([flat_info])
            info_df['ticker'] = ticker
            info_df['updated_at'] = datetime.now()
            info_df['data_source'] = 'yfinance'

            # Save to database
            self._save_to_database(info_df, 'company_info')
            logger.info(f"Successfully processed company info for {ticker}")

        except Exception as e:
            logger.error(f"Error processing company info for {ticker}: {str(e)}")
            raise

    def refresh_data(self, ticker):
        """
        Refresh data for a specific ticker by deleting and replacing it.
        :param ticker: Ticker symbol.
        :param table_name: Name of the table.
        """
        try:
            # Fetch data from API
            stock = yf.Ticker(ticker)
            data = stock.info()

            if data is None or data.empty:
                logger.warning(f"No data available for {ticker} in info table.")
                return

            data = pd.DataFrame([data])
            data['ticker'] = ticker
            data['updated_at'] = datetime.now()
            data['data_source'] = 'yfinance'
            
            # Replace data in the database
            self._delete_existing_data('company_info', ticker)
            self._save_to_database(data, 'company_info')

        except Exception as e:
            logger.error(f"Error refreshing data for {ticker} in company_info: {e}")
            raise
