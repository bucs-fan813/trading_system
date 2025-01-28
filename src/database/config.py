# trading_system/src/database/config.py

from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str
    max_retries: int = 3
    pool_size: int = 5
    max_overflow: int = 10

    @staticmethod
    def default():
        """Generate a default configuration with the database in the 'data' folder."""
        base_path = Path(__file__).resolve().parent.parent / "data"
        base_path.mkdir(exist_ok=True)  # Create 'data' directory if it doesn't exist
        database_path = base_path / "trading_system.db"
        return DatabaseConfig(url=f"sqlite:///{database_path}")
