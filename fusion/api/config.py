"""
Application configuration using pydantic-settings.

Settings can be overridden via environment variables with FUSION_GUI_ prefix.
Example: FUSION_GUI_PORT=9000 fusion gui
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server
    host: str = "127.0.0.1"
    port: int = 8765
    debug: bool = False

    # Database
    database_url: str = "sqlite:///data/gui_runs/runs.db"

    # Paths
    runs_dir: Path = Path("data/gui_runs")
    templates_dir: Path = Path("fusion/configs/templates")

    # Limits
    max_concurrent_runs: int = 1
    max_log_size_bytes: int = 10 * 1024 * 1024  # 10MB

    class Config:
        """Pydantic settings configuration."""

        env_prefix = "FUSION_GUI_"


settings = Settings()
