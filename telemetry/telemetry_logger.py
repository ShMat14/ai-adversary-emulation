# telemetry/telemetry_logger.py

import time
import json
from typing import Any, Dict
from pathlib import Path
from dataclasses import asdict

class TelemetryLogger:
    def __init__(self, output_dir: str = "./results/telemetry"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_events = []
        self.episode_id = None

    def start_episode(self):
        self.episode_events = []
        self.episode_id = int(time.time() * 1000)

    def log_event(self, action: str, state_snapshot: Any, reward: float, step: int):
        event = {
            "episode_id": self.episode_id,
            "timestamp": time.time(),
            "step": step,
            "action": action,
            "reward": reward,
            "state": {
                host: asdict(h) for host, h in state_snapshot.hosts.items()
            },
            "detection_triggered": state_snapshot.detection_triggered
        }
        self.episode_events.append(event)

    def end_episode(self):
        if not self.episode_events:
            return
        out_file = self.output_dir / f"episode_{self.episode_id}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(self.episode_events, f, indent=2)
