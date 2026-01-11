"""Dr.Case — Завантаження конфігурації"""
import yaml
from pathlib import Path
from dataclasses import asdict
from .settings import DrCaseConfig


def save_yaml(config: DrCaseConfig, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: DrCaseConfig, path: str) -> None:
    save_yaml(config, path)


def load_config(path: str) -> dict:
    return load_yaml(path)