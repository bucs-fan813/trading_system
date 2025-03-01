# trading_system/src/strategies/base_strat.py

from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Dict, Optional, Union, List, Any
from sqlalchemy import text, bindparam
from time import perf_counter

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

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict[str, Any]] = None):
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
        # Initialize cache dictionaries for company_info and financials if needed.
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    def generate_signals(self, ticker: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Abstract method to generate trading signals for a given ticker.
        New optional arguments added to align with the child class.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            initial_position (int): Starting position (0, 1, or -1).
            latest_only (bool): If True, returns only the most recent signal.

        Returns:
            pd.DataFrame: DataFrame with signals and relevant data.
        """
        pass

    def get_historical_prices(
        self,
        tickers: Union[str, List[str]],
        lookback: int = 252,
        data_source: str = 'yfinance',
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical price data from the database with optional
        date filtering and vectorized ticker support.

        If a single ticker is provided (as a string), returns a DataFrame
        indexed by date. If a list of tickers is provided, returns a combined
        DataFrame with a MultiIndex (ticker, date).

        Args:
            tickers (Union[str, List[str]]): Stock ticker symbol or list of tickers.
            lookback (int): Number of records to retrieve when no date filters are provided.
            data_source (str): Source of the data (e.g., 'yfinance').
            from_date (str, optional): Start date (inclusive) in YYYY-MM-DD format.
            to_date (str, optional): End date (inclusive) in YYYY-MM-DD format.

        Returns:
            pd.DataFrame: Historical price data.
        """
        t0 = perf_counter()
        if isinstance(tickers, str):
            params = {
                'ticker': tickers,
                'data_source': data_source
            }
            base_query = """
                SELECT date, open, high, low, close, volume 
                FROM daily_prices
                WHERE ticker = :ticker 
                AND data_source = :data_source
            """
            if from_date:
                base_query += " AND date >= :from_date"
                params['from_date'] = from_date
            if to_date:
                base_query += " AND date <= :to_date"
                params['to_date'] = to_date

            if not from_date and not to_date:
                base_query += " ORDER BY date DESC LIMIT :lookback"
                params['lookback'] = lookback
            else:
                base_query += " ORDER BY date ASC"

            query = text(base_query)
            df = self._execute_query(query, params, index_col='date')
            if not from_date and not to_date:
                df = df.sort_index()
            df.index = pd.to_datetime(df.index)
        else:
            params = {
                'tickers': tickers,
                'data_source': data_source
            }
            base_query = """
                SELECT date, open, high, low, close, volume, ticker
                FROM daily_prices
                WHERE ticker IN :tickers
                AND data_source = :data_source
            """
            if from_date:
                base_query += " AND date >= :from_date"
                params['from_date'] = from_date
            if to_date:
                base_query += " AND date <= :to_date"
                params['to_date'] = to_date

            if from_date or to_date:
                base_query += " ORDER BY ticker, date ASC"
                query = text(base_query).bindparams(bindparam("tickers", expanding=True))
                df = self._execute_query(query, params)
            else:
                base_query += " ORDER BY ticker, date ASC"
                query = text(base_query).bindparams(bindparam("tickers", expanding=True))
                df_full = self._execute_query(query, params)
                df = df_full.groupby('ticker', group_keys=False).apply(lambda group: group.iloc[-lookback:])
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index(['ticker', 'date']).sort_index()
        t1 = perf_counter()
        self.logger.debug("get_historical_prices executed in %.4f seconds", t1 - t0)
        return df

    def get_company_info(self, tickers: Union[str, List[str]], data_source: str = 'yfinance') -> Union[pd.Series, pd.DataFrame]:
        """
        Retrieve fundamental company information from the database.
        Supports a single ticker or multiple tickers. Implements simple caching for repeated queries.

        Args:
            tickers (Union[str, List[str]]): Stock ticker symbol or list of tickers.
            data_source (str): Data source (e.g., 'yfinance').

        Returns:
            pd.Series: If a single ticker is provided.
            pd.DataFrame: If multiple tickers are provided, with ticker set as the index.

        Raises:
            DataRetrievalError: If no information is found for the provided ticker(s).
        """
        cache_key = f"company_info_{tickers}_{data_source}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if isinstance(tickers, str):
            query = text("""
                SELECT *
                FROM company_info
                WHERE ticker = :ticker 
                AND data_source = :data_source         
                ORDER BY updated_at DESC 
                LIMIT 1
            """)
            df = self._execute_query(query, {'ticker': tickers, 'data_source': data_source})
            if df.empty:
                self.logger.error("No company info found for %s", tickers)
                raise DataRetrievalError(f"Company info not found for {tickers}")
            result = df.iloc[0]
        else:
            query = text("""
                SELECT *
                FROM company_info
                WHERE ticker IN :tickers 
                AND data_source = :data_source         
                ORDER BY ticker, updated_at DESC
            """).bindparams(bindparam("tickers", expanding=True))
            df = self._execute_query(query, {'tickers': tickers, 'data_source': data_source})
            if df.empty:
                self.logger.error("No company info found for provided tickers: %s", tickers)
                raise DataRetrievalError(f"Company info not found for tickers: {tickers}")
            df_sorted = df.sort_values(by=['ticker', 'updated_at'], ascending=[True, False])
            grouped = df_sorted.groupby('ticker', as_index=False).first()
            result = grouped.set_index('ticker')
        # Cache the result in memory.
        self._cache[cache_key] = result
        return result

    def get_financials(
        self,
        tickers: Union[str, List[str]],
        statement_type: str,
        lookback: Optional[int] = None,
        data_source: str = 'yfinance'
    ) -> pd.DataFrame:
        """
        Retrieve financial statements data.
        Supports a single ticker or multiple tickers. If lookback is provided,
        limits the data to the most recent periods; otherwise, extracts all available data.
        Simple caching is applied to improve performance on repeated calls.

        Args:
            tickers (Union[str, List[str]]): Stock ticker symbol or list of tickers.
            statement_type (str): Statement type ('balance_sheet', 'income_statement', 'cash_flow').
            lookback (int, optional): Number of periods to retrieve for each ticker.
            data_source (str): Data source (e.g., 'yfinance').

        Returns:
            pd.DataFrame: Financial statement data with appropriate ordering.
                         For a single ticker, the DataFrame is indexed by date.
                         For multiple tickers, the DataFrame has a MultiIndex (ticker, date).

        Raises:
            ValueError: If the statement type is invalid.
        """
        cache_key = f"financials_{tickers}_{statement_type}_{lookback}_{data_source}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        table_map = {
            'balance_sheet': 'balance_sheet',
            'income_statement': 'income_statement',
            'cash_flow': 'cash_flow'
        }
        if statement_type not in table_map:
            raise ValueError(f"Invalid statement type: {statement_type}. Valid options: {list(table_map.keys())}")
        
        if isinstance(tickers, str):
            if lookback is not None:
                query = text(f"""
                    SELECT *
                    FROM {table_map[statement_type]}
                    WHERE ticker = :ticker
                    AND data_source = :data_source
                    ORDER BY date DESC 
                    LIMIT :lookback
                """)
                params = {'ticker': tickers, 'data_source': data_source, 'lookback': lookback}
            else:
                query = text(f"""
                    SELECT *
                    FROM {table_map[statement_type]}
                    WHERE ticker = :ticker
                    AND data_source = :data_source
                    ORDER BY date ASC
                """)
                params = {'ticker': tickers, 'data_source': data_source}
            df = self._execute_query(query, params)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            else:
                self.logger.warning("No financial data retrieved for ticker: %s", tickers)
        else:
            params = {'tickers': tickers, 'data_source': data_source}
            if lookback is not None:
                base_query = f"""
                    SELECT *
                    FROM {table_map[statement_type]}
                    WHERE ticker IN :tickers
                    AND data_source = :data_source
                    ORDER BY date DESC
                """
                query = text(base_query).bindparams(bindparam("tickers", expanding=True))
                df_all = self._execute_query(query, params)
                df = df_all.groupby('ticker', group_keys=False).apply(
                    lambda group: group.sort_values(by='date', ascending=False).head(lookback).sort_values(by='date')
                )
            else:
                base_query = f"""
                    SELECT *
                    FROM {table_map[statement_type]}
                    WHERE ticker IN :tickers
                    AND data_source = :data_source
                    ORDER BY date ASC
                """
                query = text(base_query).bindparams(bindparam("tickers", expanding=True))
                df = self._execute_query(query, params)
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                if 'ticker' in df.columns:
                    df = df.set_index(['ticker', 'date']).sort_index()
        self._cache[cache_key] = df
        return df

    def _execute_query(self, query: Any, params: dict, index_col: Optional[str] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return the results in a DataFrame.
        Logs the execution time for performance monitoring.

        Args:
            query: SQLAlchemy text query.
            params (dict): Query parameters.
            index_col (str, optional): Column name to be used as the DataFrame index.

        Returns:
            pd.DataFrame: DataFrame containing query results.
        """
        t0 = perf_counter()
        try:
            with self.db_engine.connect() as connection:
                result = connection.execute(query, params)
                data = result.fetchall()
                df = pd.DataFrame(data, columns=result.keys())
                if index_col and index_col in df.columns:
                    df[index_col] = pd.to_datetime(df[index_col])
                    df = df.set_index(index_col).sort_index()
        except Exception as e:
            self.logger.error("Database error: %s", str(e))
            raise
        t1 = perf_counter()
        self.logger.debug("Executed query in %.4f seconds", t1 - t0)
        return df

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
                self.logger.error("Error disposing database engine: %s", str(e))