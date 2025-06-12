# trading_system/src/database/config.py

from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class DatabaseConfig:
    """
    Encapsulates database configuration settings.

    This dataclass holds all necessary parameters for establishing a database
    connection, including the URL and connection pool settings.

    Attributes:
        url (str): The database connection URL (e.g., 'sqlite:///path/to/db.sqlite').
        max_retries (int): The maximum number of retries for database operations.
        pool_size (int): The number of connections to keep open in the connection pool.
        max_overflow (int): The number of connections that can be opened beyond the pool_size.
    """
    url: str
    max_retries: int = field(default=3, metadata={"validate": lambda x: x > 0})
    pool_size: int = field(default=5, metadata={"validate": lambda x: x >= 1})
    max_overflow: int = field(default=10, metadata={"validate": lambda x: x >= 0})

    @staticmethod
    def default():
        """
        Creates a default DatabaseConfig for a SQLite database.

        The database file is stored in a 'data' directory at the project root.
        This method ensures the 'data' directory exists.

        Returns:
            DatabaseConfig: A default configuration instance for SQLite.
        """
        # Assumes the project root is two levels up from this file's directory
        base_path = Path(__file__).resolve().parent.parent.parent / "data"
        base_path.mkdir(exist_ok=True)
        database_path = base_path / "trading_system.db"
        return DatabaseConfig(url=f"sqlite:///{database_path}")