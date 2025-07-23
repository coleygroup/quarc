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
    "models_root": "~/quarc/models",
    "checkpoints_root": "~/quarc/checkpoints",
    "pistachio_density_path": None,
    "pistachio_namerxn_path": None,
}


def _expand(path: str | None) -> str | None:
    return None if path is None else str(Path(path).expanduser())


class Settings(dict):
    def __getattr__(self, item):  # type: ignore[override]
        return self[item]

    @property
    def data_dir(self) -> Path:
        return Path(self.data_root).expanduser()

    @property
    def processed_data_dir(self) -> Path:
        return Path(self.processed_data_root).expanduser()

    @property
    def models_dir(self) -> Path:
        return Path(self.models_root).expanduser()

    @property
    def checkpoints_dir(self) -> Path:
        return Path(self.checkpoints_root).expanduser()


def load(user_cfg: str | None = None) -> Settings:
    cfg = _DEFAULTS.copy()

    # 1) YAML override
    if user_cfg is None:
        user_cfg = os.getenv(
            "QUARC_CONFIG", str(Path.home() / ".quarc" / "configs" / "quarc_config.yaml")
        )
    p = Path(user_cfg).expanduser()
    if p.is_file():
        cfg.update(yaml.safe_load(p.read_text()))

    # 2) ENV override (UPPER-CASE names)
    for key in _DEFAULTS:
        env_val = os.getenv(key.upper())
        if env_val is not None:
            cfg[key] = env_val

    for k, v in cfg.items():
        cfg[k] = _expand(v) if isinstance(v, str) else v

    return Settings(cfg)
