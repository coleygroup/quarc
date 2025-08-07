from pathlib import Path
import os
import yaml
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_DEFAULTS: dict[str, Any] = {
    "data_root": "~/quarc/data",
    "processed_data_root": "~/quarc/data/processed",
    "checkpoints_root": "~/quarc/checkpoints",
    "logs_root": "~/quarc/logs",
    "raw_data_path": "~/quarc/data/raw",
    "pistachio_density_path": None,
    "pistachio_namerxn_path": None,
}


def _expand_path(path: str | None) -> str | None:
    return None if path is None else str(Path(path).expanduser().resolve())


class Settings(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"cfg has no attribute '{item}'")

    @property
    def data_dir(self) -> Path:
        return Path(self.data_root).expanduser().resolve()

    @property
    def processed_data_dir(self) -> Path:
        return Path(self.processed_data_root).expanduser().resolve()

    @property
    def checkpoints_dir(self) -> Path:
        return Path(self.checkpoints_root).expanduser().resolve()

    @property
    def logs_dir(self) -> Path:
        return Path(self.logs_root).expanduser().resolve()


def load() -> Settings:
    """Load QUARC path configs allow override by env vars"""
    cfg = _DEFAULTS.copy()

    # 1. override with quarc_config.yaml
    config_path = Path(__file__).parent.parent.parent / "configs" / "quarc_config.yaml"

    if config_path.is_file():
        try:
            yaml_cfg = yaml.safe_load(config_path.read_text())
            if yaml_cfg:
                cfg.update(yaml_cfg)
        except (yaml.YAMLError, OSError) as e:
            import warnings

            warnings.warn(f"Could not load config from {config_path}: {e}")

    # 2. override with env vars
    for key in _DEFAULTS:
        env_val = os.getenv(key.upper())
        if env_val is not None:
            cfg[key] = env_val

    # 3. expand paths (resolve ~ and make absolute)
    for key, value in cfg.items():
        cfg[key] = _expand_path(value) if isinstance(value, str) else value
    return Settings(cfg)
