# trading_system/tests/database/test_config.py

import pytest
from pathlib import Path
from src.database.config import DatabaseConfig

def test_default_config_creation(tmp_path):
    """Test default configuration creates correct path and settings."""
    config = DatabaseConfig.default()
    assert "trading_system.db" in config.url
    assert config.max_retries == 3
    assert config.pool_size == 5
    assert config.max_overflow == 10

def test_config_validation():
    """Test configuration field validation."""
    with pytest.raises(ValueError):
        DatabaseConfig(url="sqlite:///test.db", max_retries=-1)
        
    with pytest.raises(ValueError):
        DatabaseConfig(url="sqlite:///test.db", pool_size=0)

    valid_config = DatabaseConfig(url="sqlite:///test.db")
    assert valid_config.max_overflow == 10