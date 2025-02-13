# trading_system/src/strategies/base_strat.py

from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Dict, Optional
from sqlalchemy import text

from src.database.config import DatabaseConfig
from src.database.engine import create_db_engine

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies

    Attributes:
        db_config (DatabaseConfig): Database configuration settings
        params (dict): Strategy-specific parameters
        logger (logging.Logger): Configured logger instance
        db_engine: SQLAlchemy engine instance for database connection
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize base strategy

        Args:
            db_config: Database configuration settings
            params: Strategy-specific parameters
        """
        self.db_config = db_config
        self.params = params or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize SQLAlchemy database engine
        self.db_engine = create_db_engine(db_config)

    @abstractmethod
    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Abstract method to generate trading signals

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with signals and relevant data
        """
        pass

    def get_historical_prices(self, ticker: str, lookback: int = 252) -> pd.DataFrame:
        """
        Retrieve historical price data

        Args:
            ticker: Stock ticker symbol
            lookback: Number of trading days to look back

        Returns:
            DataFrame with historical price data
        """
        query = text(f"""
            SELECT date, open, high, low, close, volume 
            FROM daily_price_data 
            WHERE ticker = :ticker 
            ORDER BY date DESC 
            LIMIT {lookback}
        """)
        return self._execute_query(query, {'ticker': ticker}, index_col='date')

    def get_company_info(self, ticker: str) -> pd.Series:
        """
        Retrieve fundamental company information

        Args:
            ticker: Stock ticker symbol

        Returns:
            Series with company information
        """
        query = text("""
            SELECT * FROM company_info 
            WHERE ticker = :ticker 
            ORDER BY updated_at DESC 
            LIMIT 1
        """)
        df = self._execute_query(query, {'ticker': ticker})
        return df.iloc[0]

    def get_financials(self, ticker: str, statement_type: str, lookback: int = 4) -> pd.DataFrame:
        """
        Retrieve financial statements

        Args:
            ticker: Stock ticker symbol
            statement_type: Type of statement ('balance_sheet', 'income_statement', 'cash_flow')
            lookback: Number of quarters to look back

        Returns:
            DataFrame with financial statement data
        """
        table_map = {
            'balance_sheet': 'balance_sheet_data',
            'income_statement': 'income_statement_data',
            'cash_flow': 'cash_flow_data'
        }
        query = text(f"""
            SELECT * FROM {table_map[statement_type]} 
            WHERE ticker = :ticker 
            ORDER BY date DESC 
            LIMIT {lookback}
        """)
        return self._execute_query(query, {'ticker': ticker})

    def _execute_query(self, query, params: dict, index_col: Optional[str] = None) -> pd.DataFrame:
        """
        Execute database query and return results as DataFrame

        Args:
            query: SQLAlchemy text query
            params: Query parameters as a dictionary
            index_col: Column to set as index

        Returns:
            DataFrame with query results
        """
        try:
            with self.db_engine.connect() as connection:
                result = connection.execute(query, params)
                data = result.fetchall()
                df = pd.DataFrame(data, columns=result.keys())

                if index_col and index_col in df.columns:
                    df = df.set_index(index_col).sort_index()

                return df

        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
            raise

    def _validate_data(self, df: pd.DataFrame, min_records: int = 1) -> bool:
        """
        Validate retrieved data

        Args:
            df: DataFrame to validate
            min_records: Minimum required records

        Returns:
            True if data is valid
        """
        if df.empty or len(df) < min_records:
            self.logger.warning("Insufficient data for analysis")
            return False
        return True

    def __del__(self):
        """Clean up database engine resources"""
        if hasattr(self, 'db_engine'):
            self.db_engine.dispose()