# trading_system/src/strategies/base_strat.py

from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Dict, Optional, Union, List
from sqlalchemy import text, bindparam

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
        if isinstance(tickers, str):
            # Single ticker: use equality in the WHERE clause.
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

            # If no date filters are provided, then apply a SQL LIMIT
            if not from_date and not to_date:
                base_query += " ORDER BY date DESC LIMIT :lookback"
                params['lookback'] = lookback
            else:
                base_query += " ORDER BY date ASC"

            query = text(base_query)
            df = self._execute_query(query, params, index_col='date')
            # If no date filters were provided, sort the DataFrame in ascending order.
            if not from_date and not to_date:
                df = df.sort_index()
            df.index = pd.to_datetime(df.index)
            return df

        else:
            # Multiple tickers: use an IN clause.
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

            # When date filters are provided, the SQL query already limits the output.
            if from_date or to_date:
                base_query += " ORDER BY ticker, date ASC"
                query = text(base_query).bindparams(bindparam("tickers", expanding=True))
                df = self._execute_query(query, params)
            else:
                # Without date filters, we cannot apply per-ticker limit via SQL easily.
                # So, retrieve all rows ordered by ticker and date and then group by ticker
                # to keep only the last 'lookback' records for each ticker.
                base_query += " ORDER BY ticker, date ASC"
                query = text(base_query).bindparams(bindparam("tickers", expanding=True))
                df_full = self._execute_query(query, params)
                df = df_full.groupby('ticker', group_keys=False).apply(lambda group: group.iloc[-lookback:])

            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index(['ticker', 'date']).sort_index()
            return df

    def get_company_info(self, tickers: Union[str, List[str]], data_source: str = 'yfinance') -> Union[pd.Series, pd.DataFrame]:
        """
        Retrieve fundamental company information from the database.
        Supports a single ticker or multiple tickers.

        Args:
            tickers (Union[str, List[str]]): Stock ticker symbol or list of tickers.
            data_source (str): Data source (e.g., 'yfinance').

        Returns:
            pd.Series: If a single ticker is provided.
            pd.DataFrame: If multiple tickers are provided, with ticker set as the index.

        Raises:
            DataRetrievalError: If no information is found for the provided ticker(s).
        """
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
                self.logger.error(f"No company info found for {tickers}")
                raise DataRetrievalError(f"Company info not found for {tickers}")
            return df.iloc[0]
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
                self.logger.error(f"No company info found for provided tickers: {tickers}")
                raise DataRetrievalError(f"Company info not found for tickers: {tickers}")
            # For each ticker, keep only the most recent record.
            df_sorted = df.sort_values(by=['ticker', 'updated_at'], ascending=[True, False])
            grouped = df_sorted.groupby('ticker', as_index=False).first()
            return grouped.set_index('ticker')

    def get_financials(
        self,
        tickers: Union[str, List[str]],
        statement_type: str,
        lookback: int = 4,
        data_source: str = 'yfinance'
    ) -> pd.DataFrame:
        """
        Retrieve financial statements data.
        Supports a single ticker or multiple tickers.

        Args:
            tickers (Union[str, List[str]]): Stock ticker symbol or list of tickers.
            statement_type (str): Statement type ('balance_sheet', 'income_statement', 'cash_flow').
            lookback (int): Number of periods to retrieve for each ticker.
            data_source (str): Data source (e.g., 'yfinance').

        Returns:
            pd.DataFrame: Financial statement data. For single ticker, the DataFrame is indexed by date.
                          For multiple tickers, the DataFrame has a MultiIndex [ticker, date].

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

        if isinstance(tickers, str):
            query = text(f"""
                SELECT *
                FROM {table_map[statement_type]}
                WHERE ticker = :ticker
                AND data_source = :data_source
                ORDER BY date DESC 
                LIMIT :lookback
            """)
            params = {'ticker': tickers, 'data_source': data_source, 'lookback': lookback}
            df = self._execute_query(query, params)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            return df
        else:
            # When handling multiple tickers, fetch all data (ordered descending by date)
            # and then group by ticker to take the most recent 'lookback' periods.
            query_str = f"""
                SELECT *
                FROM {table_map[statement_type]}
                WHERE ticker IN :tickers
                AND data_source = :data_source
                ORDER BY date DESC
            """
            query = text(query_str).bindparams(bindparam("tickers", expanding=True))
            df_all = self._execute_query(query, {'tickers': tickers, 'data_source': data_source})
            if df_all.empty:
                self.logger.warning("No financial data retrieved for the provided tickers.")
                return df_all

            def process_group(group: pd.DataFrame) -> pd.DataFrame:
                group_sorted = group.sort_values(by='date', ascending=False).head(lookback)
                # Sort the limited group in ascending order before returning.
                return group_sorted.sort_values(by='date')

            df_processed = df_all.groupby('ticker', group_keys=False).apply(process_group)
            if not df_processed.empty:
                df_processed['date'] = pd.to_datetime(df_processed['date'])
                df_processed = df_processed.set_index(['ticker', 'date']).sort_index()
            return df_processed

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
                    # Convert the index column into datetime for proper sorting.
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