# trading_system/src/collectors/base_collector.py

from typing import Callable, Optional, Dict, Any
from datetime import datetime
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
from abc import ABC, abstractmethod
import logging
from contextlib import contextmanager
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """Base class for collecting financial data."""

    def __init__(self, db_engine):
        """
        Initialize collector with database engine and configuration.
        
        Args:
            db_engine: SQLAlchemy engine object
        """
        self.engine = db_engine
        
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
        query = text(f"DELETE FROM {table_name} WHERE ticker = :ticker")
        
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
        query = text(f"SELECT MAX(date) FROM {table_name} WHERE ticker = :ticker")
        
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
                    chunksize=1000
                )
                conn.commit()
                
            logger.info(f"Successfully saved {len(df)} rows to {table_name}")
            
        except Exception as e:
            logger.error(f"Error saving data to {table_name}: {e}")
            raise

    def _ensure_table_schema(self, table_name: str, data: pd.DataFrame) -> None:
        """
        Ensure the database table schema matches the DataFrame columns.

        Args:
            table_name: Name of the database table.
            data: DataFrame to check columns against the table.
        """
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            if not inspector.has_table(table_name):
                # If the table doesn't exist, create it based on the DataFrame schema
                data.head(0).to_sql(table_name, conn, index=False, if_exists='replace')
                return

            # Fetch existing columns
            existing_columns = {col['name'] for col in inspector.get_columns(table_name)}
            new_columns = [col for col in data.columns if col not in existing_columns]

            # Add missing columns
            for column in new_columns:
                col_type = 'TEXT'  # Default type for new columns
                if pd.api.types.is_integer_dtype(data[column]):
                    col_type = 'INTEGER'
                elif pd.api.types.is_float_dtype(data[column]):
                    col_type = 'FLOAT'
                elif pd.api.types.is_datetime64_any_dtype(data[column]):
                    col_type = 'DATETIME'

                # Add the column to the table
                alter_query = text(f"ALTER TABLE {table_name} ADD COLUMN {column} {col_type}")
                conn.execute(alter_query)

            conn.commit()

    @abstractmethod
    def refresh_data(self, ticker: str, **kwargs) -> None:
        """
        Refresh data for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            **kwargs: Additional arguments specific to each collector
        """
        pass