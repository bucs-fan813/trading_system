import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List

import pandas as pd
from sqlalchemy import inspect, text, DDL
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """
    Abstract base class for data collectors.

    Provides core functionalities for database interactions, including connection
    handling, data validation, schema management, and saving data with retries.
    """

    def __init__(self, db_engine: Engine):
        self.engine = db_engine

    @contextmanager
    def _db_connection(self):
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        if df is None or df.empty:
            return False
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True

    def _ensure_table_schema(self, table_name: str, data: pd.DataFrame) -> None:
        with self.engine.connect() as conn:
            inspector = inspect(conn)

            if not inspector.has_table(table_name):
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
                    columns.append(f'"{col}" {col_type}')
                create_table = DDL(
                    f"""CREATE TABLE IF NOT EXISTS {table_name} (
                        {', '.join(columns)}
                    )"""
                )
                try:
                    conn.execute(create_table)
                    conn.commit()
                    logger.info(f"Created table {table_name} with IF NOT EXISTS")
                except SQLAlchemyError as e:
                    logger.error(f"Error creating table {table_name}: {e}")
                    raise
                return

            existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
            missing_columns = [col for col in data.columns if col not in existing_columns]

            if missing_columns:
                preparer = self.engine.dialect.identifier_preparer
                for column in missing_columns:
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
                    except OperationalError as e:
                        if "duplicate column" in str(e).lower():
                            logger.warning(f"Column {column} already exists in {table_name}, skipping.")
                        else:
                            logger.error(f"Error adding column {column}: {e}")
                            raise
                conn.commit()

    def _get_latest_date(self, table_name: str, ticker: str) -> Optional[datetime]:
        try:
            with self._db_connection() as conn:
                if not inspect(conn).has_table(table_name):
                    logger.warning(f"Table '{table_name}' does not exist. Cannot get latest date.")
                    return None

                query = text(f"SELECT MAX(date) FROM {table_name} WHERE ticker = :ticker")
                result = conn.execute(query, {"ticker": ticker}).scalar()
                return pd.to_datetime(result) if result else None
        except OperationalError as e:
            if "no such table" in str(e).lower():
                return None
            logger.error(f"Operational error while getting latest date for {ticker}: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error while getting latest date for {ticker}: {e}")
            raise

    def _delete_existing_data(self, table_name: str, ticker: str) -> None:
        try:
            with self._db_connection() as conn:
                if inspect(conn).has_table(table_name):
                    query = text(f"DELETE FROM {table_name} WHERE ticker = :ticker")
                    transaction = conn.begin()
                    conn.execute(query, {"ticker": ticker})
                    transaction.commit()
                    logger.debug(f"Deleted existing data for {ticker} in {table_name}")
        except SQLAlchemyError as e:
            logger.error(f"Error deleting data for {ticker} in {table_name}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(OperationalError)
    )
    def _save_to_database(self, df: pd.DataFrame, table_name: str, required_columns: List[str]) -> None:
        if not self._validate_dataframe(df, required_columns):
            raise ValueError(f"Invalid data for {table_name}")

        df = df.replace([pd.NA, pd.NaT], None)
        try:
            with self._db_connection() as conn:
                df.to_sql(
                    name=table_name,
                    con=conn,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=1000
                )
            logger.debug(f"Saved {len(df)} rows to {table_name}")
        except Exception as e:
            logger.error(f"Error saving data to {table_name}: {e}")
            raise

    @abstractmethod
    def refresh_data(self, ticker: str) -> None:
        """Deletes all data for a ticker and replaces with fresh data."""
        pass

    @abstractmethod
    def fetch_and_save(self, ticker: str) -> None:
        """Fetches incremental data for a ticker and appends to existing data."""
        pass
