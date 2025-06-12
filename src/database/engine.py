# trading_system/src/database/engine.py

import logging
from sqlalchemy import create_engine, exc, event, text
from sqlalchemy.pool import NullPool
from .config import DatabaseConfig

logger = logging.getLogger(__name__)

def create_db_engine(config: DatabaseConfig):
    """
    Creates a SQLAlchemy database engine with optimized settings for SQLite.

    This function configures the engine with specific PRAGMA settings to
    enhance concurrency and performance for SQLite, which is crucial for a
    multi-threaded application. It also performs a simple health check to 
    validate connections.

    Args:
        config (DatabaseConfig): The database configuration object.

    Returns:
        sqlalchemy.engine.Engine: A configured SQLAlchemy engine instance.
    """
    try:
        engine = create_engine(
            config.url,
            # NullPool is used because SQLite's connection management is simple
            # and we handle threading concerns with PRAGMAs.
            poolclass=NullPool,
            # `check_same_thread` is disabled to allow access from multiple threads.
            # The `timeout` tells SQLite to wait if the database is locked.
            connect_args={"check_same_thread": False, "timeout": 60},
        )

        # Attach an event listener to set SQLite PRAGMAs for each new connection.
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Sets performance and concurrency-related PRAGMAs for SQLite."""
            cursor = dbapi_connection.cursor()
            # WAL (Write-Ahead Logging) mode allows concurrent reads while writing.
            cursor.execute("PRAGMA journal_mode=WAL;")
            # The busy_timeout tells SQLite how long to wait (in milliseconds)
            # if the database is locked by another process, before timing out.
            cursor.execute("PRAGMA busy_timeout=60000;")  # 60 seconds
            cursor.close()

        # Health check: Validate each connection on creation.
        @event.listens_for(engine, "engine_connect")
        def ping_connection(connection, branch):
            """Performs a health check by pinging the database."""
            if branch:  # Skip validation for sub-transactions.
                return
            try:
                connection.scalar(text("SELECT 1"))
            except exc.DBAPIError as e:
                if e.connection_invalidated:
                    logger.error("Database connection was invalidated.")
                    raise
                else:
                    logger.error("Failed to validate database connection.")
                    raise

        logger.info("Database engine created and configured successfully.")
        return engine

    except exc.SQLAlchemyError as e:
        logger.error(f"Failed to create database engine: {e}")
        raise
