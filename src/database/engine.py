# trading_system/src/database/engine.py

from sqlalchemy import create_engine, exc
import logging

logger = logging.getLogger(__name__)

def create_db_engine(config):
    """Create a database engine."""
    try:
        engine = create_engine(
            config.url,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow
        )
        logger.info("Database connection established successfully")
        return engine
    except exc.SQLAlchemyError as e:
        logger.error(f"Failed to establish database connection: {e}")
        raise
