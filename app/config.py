import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "dog-detector"
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class Config:
    roi_points: list[list[int]] = field(default_factory=list)
    enter_script: str = ""
    leave_script: str = ""
    camera_device: int = 0
    inference_interval: int = 5
    confidence: float = 0.4
    cooldown: float = 5.0
    web_port: int = 8000
    enter_frames: int = 3
    leave_frames: int = 5

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()
