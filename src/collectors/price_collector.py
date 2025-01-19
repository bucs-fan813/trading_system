# trading_system/collectors/price_collector.py

import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from src.collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)

class PriceCollector(BaseCollector):
    """Collect price data using yfinance and store it in the database."""

    def fetch_price_data(self, ticker: str):
        """
        Fetch daily prices for a ticker and save to the database.
        :param ticker: Ticker symbol.
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
                logger.warning(f"No price data available for {ticker}.")
                return

            data = data.reset_index()
            data['ticker'] = ticker
            data['updated_at'] = datetime.now()
            data['data_source'] = 'yfinance'

            self._save_to_database(data, table_name)
        except Exception as e:
            logger.error(f"Error fetching daily prices for {ticker}: {e}")
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
            data = stock.history(period='max')

            if data is None or data.empty:
                logger.warning(f"No data available for {ticker} in daily_prices.")
                return

            data = data.reset_index()
            data['ticker'] = ticker
            data['updated_at'] = datetime.now()
            data['data_source'] = 'yfinance'

            # Replace data in the database
            self._delete_existing_data("daily_prices", ticker)
            self._save_to_database(data, "daily_prices")

        except Exception as e:
            logger.error(f"Error processing refresh price data for {ticker}: {e}")
            raise