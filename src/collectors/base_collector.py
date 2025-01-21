# trading_system/src/collectors/base_collector.py

from typing import Callable, Optional, Dict, Any
from datetime import datetime
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from abc import ABC, abstractmethod
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """Base class for collecting financial data."""

    def __init__(self, db_engine, config: Dict[str, Any]):
        """
        Initialize collector with database engine and configuration.
        
        Args:
            db_engine: SQLAlchemy engine object
            config: Dictionary containing configuration parameters
        """
        self.engine = db_engine
        self.config = config
        
    @contextmanager
    def _db_connection(self):
        """Context manager for database connections."""
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            bool: True if validation passes
        """
        if df is None or df.empty:
            return False
            
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        return True

    def _delete_existing_data(self, table_name: str, ticker: str) -> None:
        """Delete existing data for a specific ticker."""
        query = f"DELETE FROM {table_name} WHERE ticker = :ticker"
        
        try:
            with self._db_connection() as conn:
                conn.execute(query, {"ticker": ticker})
                conn.commit()
            logger.info(f"Deleted existing data for {ticker} in {table_name}")
        except SQLAlchemyError as e:
            logger.error(f"Error deleting data for {ticker} in {table_name}: {e}")
            raise

    def _get_latest_date(self, table_name: str, ticker: str) -> Optional[datetime]:
        """Get the latest date for a ticker."""
        query = f"SELECT MAX(date) FROM {table_name} WHERE ticker = :ticker"
        
        try:
            with self._db_connection() as conn:
                result = conn.execute(query, {"ticker": ticker}).scalar()
                return pd.to_datetime(result) if result else None
        except SQLAlchemyError as e:
            logger.error(f"Error getting latest date for {ticker}: {e}")
            raise

    def _save_to_database(self, df: pd.DataFrame, table_name: str, 
                         required_columns: list) -> None:
        """
        Save DataFrame to database with validation.
        
        Args:
            df: DataFrame to save
            table_name: Target table name
            required_columns: List of required columns
        """
        try:
            if not self._validate_dataframe(df, required_columns):
                raise ValueError(f"Invalid data for {table_name}")

            # Handle NaN values consistently
            df = df.replace([pd.NA, pd.NaT], None)
            
            with self._db_connection() as conn:
                df.to_sql(
                    name=table_name,
                    con=conn,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=self.config.get('chunk_size', 1000)
                )
                conn.commit()
                
            logger.info(f"Successfully saved {len(df)} rows to {table_name}")
            
        except Exception as e:
            logger.error(f"Error saving data to {table_name}: {e}")
            raise

    @abstractmethod
    def refresh_data(self, ticker: str, **kwargs) -> None:
        """
        Refresh data for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            **kwargs: Additional arguments specific to each collector
        """
        pass