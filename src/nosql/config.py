from pathlib import Path 
from typing import Any 

import yaml 
from pydantic import BaseModel

class ProjectConfig(BaseModel):
    name: str = "nosql"

class AwsConfig(BaseModel):
    region: str = "eu-central-1"
    bucket: str = ""

class PathsConfig(BaseModel):
    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    runs: Path = Path("runs")

class Settings(BaseModel):
    project: ProjectConfig = ProjectConfig()
    aws: AwsConfig = AwsConfig()
    paths: PathsConfig = PathsConfig()

def load_config(path: Path = Path("configs/default.yml")) -> Settings:
    data: dict[str, Any] = {}

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    return Settings.model_validate(data)