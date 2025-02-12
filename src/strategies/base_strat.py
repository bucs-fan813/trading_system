from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Union
import logging
from src.database.config import DatabaseConfig
from src.database.engine import create_db_engine

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    
    Attributes:
        db_config (dict): MySQL database configuration
        params (dict): Strategy-specific parameters
        logger (logging.Logger): Configured logger instance
    """
    
    def __init__(self, db_config: Dict, params: Optional[Dict] = None):
        """
        Initialize base strategy
        
        Args:
            db_config: MySQL database connection configuration
            params: Strategy-specific parameters
        """
        self.db_config = DatabaseConfig.default()
        self.db_engine = create_db_engine(db_config)
        self.db_config = db_config
        self.params = params or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        

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
        query = f"""
            SELECT date, open, high, low, close, volume 
            FROM daily_price_data 
            WHERE ticker = %s 
            ORDER BY date DESC 
            LIMIT {lookback}
        """
        return self._execute_query(query, (ticker,), index_col='date')

    def get_company_info(self, ticker: str) -> pd.Series:
        """
        Retrieve fundamental company information
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Series with company information
        """
        query = """
            SELECT * FROM company_info 
            WHERE ticker = %s 
            ORDER BY updated_at DESC 
            LIMIT 1
        """
        return self._execute_query(query, (ticker,)).iloc[0]

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
        
        query = f"""
            SELECT * FROM {table_map[statement_type]} 
            WHERE ticker = %s 
            ORDER BY date DESC 
            LIMIT {lookback}
        """
        return self._execute_query(query, (ticker,))

    def _execute_query(self, query: str, params: tuple, index_col: Optional[str] = None) -> pd.DataFrame:
        """
        Execute database query and return results as DataFrame
        
        Args:
            query: SQL query string
            params: Query parameters
            index_col: Column to set as index
            
        Returns:
            DataFrame with query results
        """
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(query, params)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            
            if index_col and index_col in df.columns:
                df = df.set_index(index_col).sort_index()
                
            return df
            
        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
            raise
        finally:
            cursor.close()

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
        """Clean up database connection"""
        if hasattr(self, 'db_connection') and self.db_connection.is_connected():
            self.db_connection.close()