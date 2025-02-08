# trading_system/src/database/engine.py


from sqlalchemy import create_engine, exc, event, text
import logging
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

def create_db_engine(config):
    """
    Create a database engine with connection validation and SQLite PRAGMAs for concurrency.
    
    Args:
        config: An object with attributes:
            - url (str): The database URL.
            - pool_size (int): Number of connections in the pool.
            - max_overflow (int): Extra connections allowed beyond the pool size.

    Returns:
        engine: A SQLAlchemy engine instance.
    """
    try:
        engine = create_engine(
            config.url,
            # Disable connection pooling since SQLite isn’t designed for high concurrency.
            poolclass=NullPool,
            # Allow multi-thread access; add a timeout so SQLite will wait before erroring.
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        
        # Set SQLite-specific PRAGMAs on each new connection.
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            # Enable WAL mode which improves concurrent read/write performance.
            cursor.execute("PRAGMA journal_mode=WAL;")
            # Increase busy timeout (in milliseconds). Adjust as needed.
            cursor.execute("PRAGMA busy_timeout=30000;")
            cursor.close()

        # Optional: Validate connection health.
        @event.listens_for(engine, "engine_connect")
        def ping_connection(connection, branch):
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

        logger.info("Database connection established successfully")
        return engine

    except exc.SQLAlchemyError as e:
        logger.error(f"Failed to establish database connection: {e}")
        raise
