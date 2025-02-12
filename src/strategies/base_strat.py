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

# Example Strategy Implementation
class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def generate_signals(self, ticker: str) -> pd.DataFrame:
        # Get price data with default 1-year lookback
        prices = self.get_historical_prices(ticker)
        
        if not self._validate_data(prices, min_records=20):
            return pd.DataFrame()
            
        # Calculate 12-month momentum
        prices['returns'] = prices['close'].pct_change(periods=21)
        prices['12m_momentum'] = prices['close'].pct_change(periods=252)
        
        # Generate signals
        prices['signal'] = 0
        prices.loc[prices['12m_momentum'] > 0.15, 'signal'] = 1  # Buy
        prices.loc[prices['12m_momentum'] < -0.15, 'signal'] = -1  # Sell
        
        return prices[['close', 'returns', '12m_momentum', 'signal']].dropna()

class ValueStrategy(BaseStrategy):
    """Fundamental value investing strategy"""
    
    def generate_signals(self, ticker: str) -> pd.DataFrame:
        # Get financial data
        balance_sheet = self.get_financials(ticker, 'balance_sheet')
        income_stmt = self.get_financials(ticker, 'income_statement')
        
        if not self._validate_data(balance_sheet) or not self._validate_data(income_stmt):
            return pd.DataFrame()
            
        # Calculate fundamental metrics
        latest_bs = balance_sheet.iloc[0]
        latest_is = income_stmt.iloc[0]
        
        metrics = {
            'pe_ratio': latest_is['currentprice'] / latest_is['trailingeps'],
            'pb_ratio': latest_bs['bookvalue'] / latest_bs['marketcap'],
            'debt_to_equity': latest_bs['totaldebt'] / latest_bs['total_equity_gross_minority_interest']
        }
        
        # Generate signals based on value metrics
        signals = pd.DataFrame(index=[pd.Timestamp.today()])
        signals['value_score'] = 0
        
        if metrics['pe_ratio'] < 15:
            signals['value_score'] += 1
        if metrics['pb_ratio'] < 1:
            signals['value_score'] += 1
        if metrics['debt_to_equity'] < 0.5:
            signals['value_score'] += 1
            
        return signals