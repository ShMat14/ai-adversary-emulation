# telemetry/exporters.py

import json
from pathlib import Path
import requests  # for ELK export

def export_to_csv(json_dir: str, csv_path: str):
    import pandas as pd
    rows = []
    for jf in Path(json_dir).glob("episode_*.json"):
        data = json.loads(jf.read_text())
        for ev in data:
            rows.append({
                "episode_id": ev["episode_id"],
                "timestamp": ev["timestamp"],
                "step": ev["step"],
                "action": ev["action"],
                "reward": ev["reward"],
                "detection_triggered": ev["detection_triggered"],
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

def export_to_elasticsearch(json_dir: str, index_name: str, elk_url: str):
    """
    elk_url example: 'http://localhost:9200'
    """
    for jf in Path(json_dir).glob("episode_*.json"):
        data = json.loads(jf.read_text())
        for ev in data:
            # Basic indexing
            r = requests.post(f"{elk_url}/{index_name}/_doc", json=ev)
            r.raise_for_status()
