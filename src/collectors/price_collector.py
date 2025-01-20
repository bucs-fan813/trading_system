# trading_system/collectors/price_collector.py											  

import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from src.collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)

class PriceCollector(BaseCollector):
    """Collect price data using yfinance and store it in the database."""

    def fetch_price_data(self, ticker: str) -> None:
        """
        Fetch additional daily prices for a ticker and save to the database.

        Args:
            ticker: Ticker symbol.
        """
        try:
            table_name = "daily_prices"
            latest_date = self._get_latest_date(table_name, ticker)

            stock = yf.Ticker(ticker)

            if latest_date:
                start_date = latest_date + timedelta(days=1)
            else:
                start_date = datetime.now() - timedelta(days=365)	 
																 

            data = stock.history(start=start_date.strftime('%Y-%m-%d'))

            if data.empty:
                logger.warning(f"No new price data available for {ticker}.")
                return

            data = data.reset_index()
            data['ticker'] = ticker
            data['updated_at'] = datetime.now()
            data['data_source'] = 'yfinance'

            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker', 'updated_at', 'data_source']
            self._save_to_database(data, table_name, required_columns)

        except Exception as e:
            logger.error(f"Error fetching daily prices for {ticker}: {e}")
            raise

    def refresh_data(self, ticker: str) -> None:
        """
        Refresh data for a specific ticker by deleting all existing data and replacing it with new data.

        Args:
            ticker: Ticker symbol.
        """

        try:
            # Fetch data from API
            stock = yf.Ticker(ticker)
            data = stock.history(period='max')

            if data.empty:
                logger.warning(f"No data available for {ticker} in daily_prices.")
                return

            data = data.reset_index()
            data['ticker'] = ticker
            data['updated_at'] = datetime.now()
            data['data_source'] = 'yfinance'

            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker', 'updated_at', 'data_source']

            # Replace data in the database
            self._delete_existing_data("daily_prices", ticker)
            self._save_to_database(data, "daily_prices", required_columns)

            logger.info(f"Successfully refreshed data for {ticker} in daily_prices.")

        except Exception as e:
            logger.error(f"Error refreshing data for {ticker}: {e}")
            raise
