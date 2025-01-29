# trading_system/tests/collectors/test_base_collector.py

import pytest
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from src.collectors.base_collector import BaseCollector

@pytest.fixture
def test_engine():
    return create_engine("sqlite:///:memory:")

def test_validate_dataframe():
    """Test DataFrame validation logic."""
    collector = BaseCollector(None)
    valid_df = pd.DataFrame({"date": [datetime.now()], "value": [100]})
    assert collector._validate_dataframe(valid_df, ["date"]) is True
    
    invalid_df = pd.DataFrame({"wrong_col": [1]})
    assert collector._validate_dataframe(invalid_df, ["date"]) is False

def test_ensure_table_schema(test_engine):
    """Test schema synchronization with DataFrame."""
    collector = BaseCollector(test_engine)
    test_df = pd.DataFrame({
        "date": [datetime.now()],
        "new_col": [1.23],
        "ticker": ["TEST"]
    })
    
    collector._ensure_table_schema("test_table", test_df)
    inspector = test_engine.dialect.inspector(test_engine)
    assert "test_table" in inspector.get_table_names()
    assert "new_col" in [col["name"] for col in inspector.get_columns("test_table")]