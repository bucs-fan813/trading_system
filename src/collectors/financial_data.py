import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class FinancialDataCollector:
    """Collect financial data using yfinance and store it in the database."""

    def __init__(self, db_engine):
        """
        Initialize with a database engine.
        :param db_engine: SQLAlchemy engine object.
        """
        self.engine = db_engine

    
    def _delete_existing_data(self, table_name, ticker):
        """
        Delete existing data for a specific ticker from a specific table.
        :param table_name: Name of the table.
        :param ticker: Ticker symbol.
        """
        query = f"DELETE FROM {table_name} WHERE ticker = :ticker"
        try:
            with self.engine.connect() as conn:
                conn.execute(query, {"ticker": ticker})
            logger.info(f"Deleted existing data for {ticker} in {table_name}.")
        except SQLAlchemyError as e:
            logger.error(f"Error deleting data for {ticker} in {table_name}: {e}")
            raise

    def refresh_data(self, ticker, table_name, fetch_function):
        """
        Refresh data for a specific ticker by deleting and replacing it.
        :param ticker: Ticker symbol.
        :param table_name: Name of the table.
        :param fetch_function: Function to fetch data from yfinance.
        """
        try:
            # Fetch data from API
            stock = yf.Ticker(ticker)
            data = fetch_function(stock)

            if data is None or data.empty:
                logger.warning(f"No data available for {ticker} in {table_name}.")
                return

            # Process and save data
            if table_name == "daily_prices":
                data = data.reset_index()
                data['ticker'] = ticker
                data['updated_at'] = datetime.now()
                data['data_source'] = 'yfinance'
            elif table_name == "company_info":
                data = pd.DataFrame([data])
                data['ticker'] = ticker
                data['updated_at'] = datetime.now()
                data['data_source'] = 'yfinance'
            else:
                data = data.fillna(pd.NA).T
                data['ticker'] = ticker
                data = data.reset_index().rename(columns={'index': 'date'})
                data['updated_at'] = datetime.now()
                data['data_source'] = 'yfinance'

            # Replace data in the database
            self._delete_existing_data(table_name, ticker)
            self._save_to_database(data, table_name)

        except Exception as e:
            logger.error(f"Error refreshing data for {ticker} in {table_name}: {e}")
            raise

    def _get_latest_date(self, table_name, ticker):
        """
        Get the latest date for a ticker from a specific table.
        :param table_name: Name of the table.
        :param ticker: Ticker symbol.
        :return: Latest date as a datetime object or None.
        """
        query = f"SELECT MAX(date) FROM {table_name} WHERE ticker = :ticker"
        with self.engine.connect() as conn:
            result = conn.execute(query, {"ticker": ticker}).scalar()
            return pd.to_datetime(result) if result else None

    def _save_to_database(self, df, table_name):
        """
        Save a DataFrame to the database.
        :param df: Pandas DataFrame to save.
        :param table_name: Table name in the database.
        """
        try:
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            logger.info(f"Data saved to {table_name} successfully.")
        except SQLAlchemyError as e:
            logger.error(f"Error saving data to {table_name}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_financial_statement(self, ticker, statement_type, fetch_function):
        """
        Fetch and process financial statements.
        :param ticker: Ticker symbol.
        :param statement_type: Type of statement (e.g., 'balance_sheet').
        :param fetch_function: Function to fetch the statement from yfinance.
        """
        try:
            table_name = statement_type.lower()
            latest_date = self._get_latest_date(table_name, ticker)

            if latest_date and (datetime.now() - latest_date < timedelta(days=40)):
                logger.info(f"Skipping {ticker} {statement_type} - data is up to date.")
                return

            stock = yf.Ticker(ticker)
            data = fetch_function(stock)

            if data is None or data.empty:
                logger.warning(f"No {statement_type} data available for {ticker}.")
                return

            data = data.fillna(pd.NA).T
            data['ticker'] = ticker
            data = data.reset_index().rename(columns={'index': 'date'})

            if latest_date:
                data = data[data['date'] > latest_date]

            if not data.empty:
                data['updated_at'] = datetime.now()
                data['data_source'] = 'yfinance'
                self._save_to_database(data, table_name)

        except Exception as e:
            logger.error(f"Error processing {statement_type} for {ticker}: {e}")
            raise

    def fetch_daily_prices(self, ticker):
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

    def fetch_company_info(self, ticker):
        """
        Fetch company information and save to the database.
        :param ticker: Ticker symbol.
        """
        try:
            table_name = "company_info"

            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                logger.warning(f"No company info available for {ticker}.")
                return

            # Convert dictionary to DataFrame
            data = pd.DataFrame([info])
            data['ticker'] = ticker
            data['updated_at'] = datetime.now()
            data['data_source'] = 'yfinance'

            self._save_to_database(data, table_name)
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {e}")
            raise
