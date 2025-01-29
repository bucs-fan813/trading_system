# trading_system/tests/database/test_engine.py

import pytest
from sqlalchemy import create_engine
from src.database.engine import create_db_engine
from src.database.config import DatabaseConfig

def test_engine_creation_success():
    """Test successful engine creation with valid config."""
    config = DatabaseConfig(url="sqlite:///:memory:")
    engine = create_db_engine(config)
    assert engine.pool.size() == 5

def test_engine_connection_validation(caplog):
    """Test connection validation through engine events."""
    config = DatabaseConfig(url="sqlite:///:memory:")
    engine = create_db_engine(config)
    with engine.connect() as conn:
        conn.execute("SELECT 1")
    assert "established successfully" in caplog.text

def test_invalid_engine_creation():
    """Test engine creation failure with invalid URL."""
    config = DatabaseConfig(url="invalid://user:pass@host/db")
    with pytest.raises(Exception):
        create_db_engine(config)