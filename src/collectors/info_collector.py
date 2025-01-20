# trading_system/collectors/info_collector.py
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, List
from src.collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)

class InfoCollector(BaseCollector):
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
    def fetch_company_info(self, ticker: str) -> pd.DataFrame:
        """
        Fetch company information for a specific ticker.
        
        Args:
            ticker: Ticker symbol.
        
        Returns:
            pd.DataFrame: DataFrame containing company info.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                logger.warning(f"No company info available for {ticker}")
                return pd.DataFrame()

            # Flatten and convert to DataFrame
            flat_info = self._flatten_nested_dict(info)
            info_df = pd.DataFrame([flat_info])
            info_df['ticker'] = ticker
            info_df['updated_at'] = datetime.now()
            info_df['data_source'] = 'yfinance'
            return info_df

        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {str(e)}")
            raise

    def refresh_data(self, ticker: str, table_name: str = 'company_info', fetch_function=None):
        """
        Refresh data for a specific ticker by deleting and replacing it.
        
        Args:
            ticker: Ticker symbol.
            table_name: Target table name (default: 'company_info').
            fetch_function: Function to fetch data (optional, default: `self.fetch_company_info`).
        """
        try:
            # Use the provided fetch_function or default to fetch_company_info
            fetch_function = fetch_function or self.fetch_company_info

            # Fetch data
            data = fetch_function(ticker)

            if data.empty:
                logger.warning(f"No data available for {ticker} in {table_name}.")
                return

            # Replace data in the database
            self._delete_existing_data(table_name, ticker)
            self._save_to_database(data, table_name, required_columns=['ticker', 'updated_at', 'data_source'])

            logger.info(f"Successfully refreshed data for {ticker} in {table_name}")

        except Exception as e:
            logger.error(f"Error refreshing data for {ticker} in {table_name}: {e}")
            raise
