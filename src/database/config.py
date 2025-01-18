from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str
    max_retries: int = 3
    pool_size: int = 5
    max_overflow: int = 10
