"""Application configuration settings."""

import os
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "app" / "data"


class AppSettings(BaseModel):
    """Runtime configuration for the adaptive career pathway backend."""

    app_name: str = "Adaptive Career-Pathway Recommendation Engine"
    version: str = "0.1.0"
    environment: str = Field(default_factory=lambda: os.getenv("ENV", "development"))
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{BASE_DIR / 'career_pathway.db'}"
        )
    )
    competency_data_path: Path = DATA_DIR / "competency_v1.json"
    questions_data_path: Path = DATA_DIR / "questions_v1.json"


settings = AppSettings()
