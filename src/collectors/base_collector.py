# trading_system/src/collectors/base_collector.py

from typing import Callable, Optional, Dict, Any
from datetime import datetime
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.sql import text
from abc import ABC, abstractmethod
import logging
from contextlib import contextmanager 
from sqlalchemy import Integer, Float, String, DateTime, text, exc, inspect, DDL
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type


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
        """Context manager for database connections with retry"""
        connection = self.engine.connect()
        transaction = connection.begin()
        try:
            yield connection
            transaction.commit()
        except Exception as e:
            transaction.rollback()
            raise
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
        """
        Delete existing data for a specific ticker.
        """
        if not inspect(self.engine).has_table(table_name):
            return  # Skip deletion if table doesn't exist
        
        query = text(f"DELETE FROM {table_name} WHERE ticker = :ticker")
        
        try:
            with self._db_connection() as conn:
                conn.execute(query, {"ticker": ticker})
                # conn.commit()
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
        except exc.OperationalError as e:
            if "no such table" in str(e).lower():
                return None
            else:
                logger.error(f"Operational error getting latest date for {ticker}: {e}")
                raise
        except exc.SQLAlchemyError as e:
            logger.error(f"Error getting latest date for {ticker}: {e}")
            raise

    @retry(stop=stop_after_attempt(3),
       wait=wait_fixed(1),
       retry=retry_if_exception_type(OperationalError))
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
                # conn.commit()
                
            logger.info(f"Successfully saved {len(df)} rows to {table_name}")
            
        except Exception as e:
            logger.error(f"Error saving data to {table_name}: {e}")
            raise

    def _ensure_table_schema(self, table_name: str, data: pd.DataFrame) -> None:
        """
        Ensure the database table schema matches the DataFrame columns.
        
        This method uses an atomic table creation statement with IF NOT EXISTS
        to avoid race conditions when multiple threads attempt to create the same table.
        If the table already exists, any missing columns (based on the DataFrame) are added.
        
        Args:
            table_name: Name of the database table.
            data: DataFrame whose columns are used to verify or build the table schema.
        """
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            
            # If the table does not exist, create it atomically using IF NOT EXISTS
            if not inspector.has_table(table_name):
                # Generate column definitions for table creation
                columns = []
                for col in data.columns:
                    if pd.api.types.is_integer_dtype(data[col]):
                        col_type = "INTEGER"
                    elif pd.api.types.is_float_dtype(data[col]):
                        col_type = "FLOAT"
                    elif pd.api.types.is_datetime64_any_dtype(data[col]):
                        col_type = "DATETIME"
                    else:
                        col_type = "VARCHAR"
                    
                    # Quote column names to handle reserved words or special characters
                    columns.append(f'"{col}" {col_type}')
                
                # Construct the DDL statement using IF NOT EXISTS to avoid race conditions
                create_table = DDL(
                    f"""CREATE TABLE IF NOT EXISTS {table_name} (
                        {', '.join(columns)}
                    )"""
                )
                
                try:
                    conn.execute(create_table)
                    conn.commit()
                    logger.info(f"Created table {table_name} with IF NOT EXISTS")
                except exc.SQLAlchemyError as e:
                    logger.error(f"Error creating table {table_name}: {e}")
                    raise
                return  # Table has been created, so exit early.
            
            # If the table already exists, check for missing columns.
            existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
            # Compare using the original column names, assuming they match exactly.
            missing_columns = [col for col in data.columns if col not in existing_columns]

            if missing_columns:
                # Use the engine's identifier preparer to properly quote column names.
                preparer = self.engine.dialect.identifier_preparer
                for column in missing_columns:
                    # Determine the SQL type for the column based on the DataFrame dtype.
                    if pd.api.types.is_integer_dtype(data[column]):
                        col_type = "INTEGER"
                    elif pd.api.types.is_float_dtype(data[column]):
                        col_type = "FLOAT"
                    elif pd.api.types.is_datetime64_any_dtype(data[column]):
                        col_type = "DATETIME"
                    else:
                        col_type = "VARCHAR"
                        
                    column_quoted = preparer.quote(column)
                    alter_sql = f'ALTER TABLE {table_name} ADD COLUMN {column_quoted} {col_type}'
                    try:
                        conn.execute(text(alter_sql))
                        logger.info(f"Added column {column_quoted} to {table_name}")
                    except exc.OperationalError as e:
                        # If the error is due to a duplicate column, log a warning and continue.
                        if "duplicate column" in str(e).lower():
                            logger.warning(f"Column {column} already exists in {table_name}, skipping.")
                        else:
                            logger.error(f"Error adding column {column}: {e}")
                            raise
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