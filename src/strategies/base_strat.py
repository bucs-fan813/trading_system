# trading_system/src/strategies/base_strat.py

from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Dict, Optional
from sqlalchemy import text

from src.database.config import DatabaseConfig
from src.database.engine import create_db_engine


class DataRetrievalError(Exception):
    """Custom exception for data retrieval failures."""
    pass


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Attributes:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict): Strategy-specific parameters.
        logger (logging.Logger): Logger instance.
        db_engine: SQLAlchemy engine instance for database connection.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the base strategy.

        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Strategy-specific parameters.
        """
        self.db_config = db_config
        self.params = params or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_engine = create_db_engine(db_config)

    @abstractmethod
    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Abstract method to generate trading signals for a given ticker.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: DataFrame with signals and relevant data.
        """
        pass

    def get_historical_prices(
        self,
        ticker: str,
        lookback: int = 252,
        data_source: str = 'yfinance',
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical price data from the database with optional date filtering.

        Args:
            ticker (str): Stock ticker symbol.
            lookback (int): Number of records to retrieve if date filters are not provided.
            data_source (str): Source of the data (e.g., 'yfinance').
            from_date (str, optional): Start date (inclusive) in YYYY-MM-DD format.
            to_date (str, optional): End date (inclusive) in YYYY-MM-DD format.

        Returns:
            pd.DataFrame: Historical price data with 'date' as index.
        """
        params = {
            'ticker': ticker,
            'data_source': data_source
        }
        base_query = """
            SELECT date, open, high, low, close, volume 
            FROM daily_prices
            WHERE ticker = :ticker 
            AND data_source = :data_source
        """
        # Add date filtering if provided:
        if from_date:
            base_query += " AND date >= :from_date"
            params['from_date'] = from_date
        if to_date:
            base_query += " AND date <= :to_date"
            params['to_date'] = to_date

        # If no date filters are provided, use a lookback limit (retrieve the latest records)
        if not from_date and not to_date:
            base_query += " ORDER BY date DESC LIMIT :lookback"
            params['lookback'] = lookback
        else:
            # Order chronologically when a specific date range is provided.
            base_query += " ORDER BY date ASC"

        query = text(base_query)
        df = self._execute_query(query, params, index_col='date')

        # If no date filters were provided, the query orders by descending date;
        # sort the DataFrame in ascending order.
        if not from_date and not to_date:
            df = df.sort_index()
        return df

    def get_company_info(self, ticker: str, data_source: str = 'yfinance') -> pd.Series:
        """
        Retrieve fundamental company information from the database.

        Args:
            ticker (str): Stock ticker symbol.
            data_source (str): Data source (e.g., 'yfinance').

        Returns:
            pd.Series: Series containing company information.

        Raises:
            DataRetrievalError: If no information is found for the company.
        """
        query = text("""
            SELECT * FROM company_info 
            WHERE ticker = :ticker 
            AND data_source = :data_source         
            ORDER BY updated_at DESC 
            LIMIT 1
        """)
        df = self._execute_query(query, {'ticker': ticker, 'data_source': data_source})
        if df.empty:
            self.logger.error(f"No company info found for {ticker}")
            raise DataRetrievalError(f"Company info not found for {ticker}")
        return df.iloc[0]

    def get_financials(
        self,
        ticker: str,
        statement_type: str,
        lookback: int = 4,
        data_source: str = 'yfinance'
    ) -> pd.DataFrame:
        """
        Retrieve financial statements data.

        Args:
            ticker (str): Stock ticker symbol.
            statement_type (str): Statement type ('balance_sheet', 'income_statement', 'cash_flow').
            lookback (int): Number of periods to retrieve.
            data_source (str): Data source (e.g., 'yfinance').

        Returns:
            pd.DataFrame: Financial statement data.

        Raises:
            ValueError: If the statement type is invalid.
        """
        table_map = {
            'balance_sheet': 'balance_sheet',
            'income_statement': 'income_statement',
            'cash_flow': 'cash_flow'
        }
        if statement_type not in table_map:
            raise ValueError(f"Invalid statement type: {statement_type}. "
                             f"Valid options: {list(table_map.keys())}")

        query = text(f"""
            SELECT * FROM {table_map[statement_type]} 
            WHERE ticker = :ticker 
            AND data_source = :data_source
            ORDER BY date DESC 
            LIMIT :lookback
        """)
        return self._execute_query(query, {'ticker': ticker, 'data_source': data_source, 'lookback': lookback})

    def _execute_query(self, query, params: dict, index_col: Optional[str] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return the results in a DataFrame.

        Args:
            query: SQLAlchemy text query.
            params (dict): Query parameters.
            index_col (str, optional): Column name to be used as the DataFrame index.

        Returns:
            pd.DataFrame: DataFrame containing query results.
        """
        try:
            with self.db_engine.connect() as connection:
                result = connection.execute(query, params)
                data = result.fetchall()
                df = pd.DataFrame(data, columns=result.keys())
                if index_col and index_col in df.columns:
                    # Convert the date column into datetime for proper sorting.
                    df[index_col] = pd.to_datetime(df[index_col])
                    df = df.set_index(index_col).sort_index()
                return df
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
            raise

    def _validate_data(self, df: pd.DataFrame, min_records: int = 1) -> bool:
        """
        Validate that the DataFrame has sufficient records for analysis.

        Args:
            df (pd.DataFrame): DataFrame to check.
            min_records (int): Minimum required records.

        Returns:
            bool: True if data has enough records, False otherwise.
        """
        if df.empty or len(df) < min_records:
            self.logger.warning("Insufficient data for analysis")
            return False
        return True

    def __del__(self):
        """
        Dispose of the database engine when the strategy instance is deleted.
        """
        if hasattr(self, 'db_engine'):
            try:
                self.db_engine.dispose()
                self.logger.debug("Database engine disposed successfully")
            except Exception as e:
                self.logger.error(f"Error disposing database engine: {str(e)}")