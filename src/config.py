"""Load and snapshot JSON configs."""
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


@dataclass
class Config:
    devices: dict
    models: dict
    drift: dict
    algo: dict
    experiment: dict

    @classmethod
    def from_dir(cls, config_dir: str = "configs") -> "Config":
        d = Path(config_dir)
        return cls(
            devices=_load_json(d / "devices.json"),
            models=_load_json(d / "models.json"),
            drift=_load_json(d / "drift.json"),
            algo=_load_json(d / "algo.json"),
            experiment=_load_json(d / "experiment.json"),
        )

    def to_dict(self) -> dict:
        return {
            "devices": self.devices,
            "models": self.models,
            "drift": self.drift,
            "algo": self.algo,
            "experiment": self.experiment,
        }

    def snapshot(self, out_path: str) -> None:
        """Dump the full resolved config to disk for reproducibility."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_model(self, name: str = None) -> dict:
        name = name or self.models["default_model"]
        if name not in self.models["models"]:
            raise KeyError(f"Unknown model: {name}")
        return self.models["models"][name]


if __name__ == "__main__":
    c = Config.from_dir("configs")
    print(f"Loaded {len(c.devices['tiers'])} tiers, {len(c.models['models'])} models")
    print(f"Default model: {c.experiment['model_name']}")
    print(f"Schemes: {c.experiment['schemes']}")
