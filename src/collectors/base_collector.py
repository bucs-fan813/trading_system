# trading_system/collectors/base_collector.py

import logging
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
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

    @abstractmethod
    def refresh_data(self, ticker, table_name, fetch_function):
        """
        Refresh data for a specific ticker and store it in the database.
        This method should be implemented by subclasses.
        :param ticker: Ticker symbol.
        :param table_name: Name of the table.
        :param fetch_function: Function to fetch data.
        """
        pass

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