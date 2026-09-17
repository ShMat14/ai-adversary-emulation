# analysis/compute_metrics.py

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

import pandas as pd

# --- CONFIG ---

from pathlib import Path

TELEMETRY_DIR = Path("results/telemetry/rl")

SNORT_ALERT_FILE = Path("./results/snort/alert.fast")  # optional, can be absent

# Map abstract actions to (tactic, technique_id) pairs — synced with attack_actions.py
ACTION_TO_ATTACK = {
    "PHISHING_EMAIL":       ("Initial Access",       "T1566"),
    "BRUTE_FORCE_SSH":      ("Credential Access",    "T1110"),
    "NETWORK_SCAN":         ("Discovery",            "T1046"),
    "VALID_ACCOUNTS_LOGIN": ("Defense Evasion",      "T1078"),
    "INSTALL_BACKDOOR":     ("Persistence",          "T1543"),
    "CLEAR_LOGS":           ("Defense Evasion",      "T1070"),
    "LATERAL_MOVE_SMB":     ("Lateral Movement",     "T1021"),
    "PRIV_ESC_SUDO":        ("Privilege Escalation", "T1068"),
    "EXFILTRATE_DATA":      ("Exfiltration",         "T1041"),
    "RANSOMWARE_ENCRYPT":   ("Impact",               "T1486"),
    "SQL_INJECTION":        ("Initial Access",       "T1190"),  # Exploit Public-Facing App
}


# --- HELPER FUNCTIONS ---


def load_episodes(telemetry_dir):
    """
    Load telemetry JSONs from a directory and return a dict:
      { episode_id: [events, ...], ... }
    It is robust to a few formats:
      - {"episode_id": ..., "events": [...]}
      - {"events": [...]}
      - [...] (a raw list of events)
    """
    telemetry_dir = Path(telemetry_dir)
    episodes = {}

    if not telemetry_dir.exists():
        print(f"[!] Telemetry dir does not exist: {telemetry_dir.resolve()}")
        return episodes

    files = list(telemetry_dir.glob("*.json"))
    if not files:
        print(f"[!] No JSON files found in {telemetry_dir.resolve()}")
        return episodes

    print(f"[*] Found {len(files)} JSON files in {telemetry_dir.resolve()}")

    for jf in files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Case 1: dict with explicit episode_id + events
        if isinstance(data, dict):
            episode_id = data.get("episode_id", jf.stem)

            # try to find the list of events
            if isinstance(data.get("events"), list):
                events = data["events"]
            elif isinstance(data.get("episode_events"), list):
                events = data["episode_events"]
            else:
                # maybe the dict itself is one event; wrap it
                events = [data]

        # Case 2: top-level list = already a list of events
        elif isinstance(data, list):
            episode_id = jf.stem
            events = data

        else:
            # Unknown format, skip
            print(f"[!] Skipping {jf} (unknown JSON structure)")
            continue

        episodes[episode_id] = events

    return episodes



def compute_per_episode_stats(events: List[dict]) -> dict:
    """
    Compute stats for a single episode:
      - detected (bool)
      - detection_step (int or None)
      - actions (list of action names)
      - steps (int)
    """
    detected = False
    detection_step = None
    actions = []
    steps = 0

    for ev in events:
        steps = max(steps, ev.get("step", 0))
        actions.append(ev["action"])
        if ev.get("detection_triggered"):
            if not detected:
                detected = True
                detection_step = ev.get("step", None)

    return {
        "detected": detected,
        "detection_step": detection_step,
        "actions": actions,
        "steps": steps,
    }


def parse_snort_alerts(alert_file: Path) -> Set[int]:
    """
    Example: parse Snort 'fast' alerts and extract episode IDs if they are logged.
    For simplicity, assume each line contains 'episode_id=<id>' if relevant.
    """
    if not alert_file.exists():
        return set()

    detected_episodes = set()
    with alert_file.open("r", encoding="utf-8") as f:
        for line in f:
            if "episode_id=" in line:
                try:
                    part = line.strip().split("episode_id=")[-1]
                    eid = int(part.split()[0].strip("[];,"))
                    detected_episodes.add(eid)
                except ValueError:
                    continue
    return detected_episodes


def compute_metrics(episodes: Dict[int, List[dict]]) -> Tuple[pd.DataFrame, dict]:
    records = []
    coverage_techniques: Set[str] = set()
    coverage_tactics: Set[str] = set()
    sequences: Set[Tuple[str, ...]] = set()

    for eid, events in episodes.items():
        stats = compute_per_episode_stats(events)
        actions = stats["actions"]
        sequence = tuple(actions)
        sequences.add(sequence)

        # ATT&CK coverage
        for a in actions:
            if a in ACTION_TO_ATTACK:
                tactic, tech = ACTION_TO_ATTACK[a]
                coverage_tactics.add(tactic)
                coverage_techniques.add(tech)

        records.append({
            "episode_id": eid,
            "detected": stats["detected"],
            "detection_step": stats["detection_step"],
            "steps": stats["steps"],
            "num_actions": len(actions),
            "sequence": sequence,
        })

    df = pd.DataFrame(records)

    # Basic metrics
    num_episodes = len(df)
    detection_rate = df["detected"].mean() if num_episodes > 0 else 0.0
    avg_ttd = df["detection_step"].dropna().mean() if df["detection_step"].notna().any() else None
    scenario_diversity = len(sequences)

    metrics = {
        "num_episodes": num_episodes,
        "detection_rate": detection_rate,
        "avg_time_to_detection_steps": avg_ttd,
        "attack_coverage_num_techniques": len(coverage_techniques),
        "attack_coverage_num_tactics": len(coverage_tactics),
        "scenario_diversity": scenario_diversity,
        "unique_techniques": sorted(coverage_techniques),
        "unique_tactics": sorted(coverage_tactics),
    }

    return df, metrics


# --- MAIN ---

def main():
    episodes = load_episodes(TELEMETRY_DIR)
    if not episodes:
        print("No telemetry episodes found.")
        return

    df, metrics = compute_metrics(episodes)

    print("=== Episode-level stats ===")
    print(df[["episode_id", "detected", "detection_step", "steps", "num_actions"]])

    print("\n=== Aggregate metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Save for thesis plots
    out_dir = Path("./results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "episodes_stats.csv", index=False)
    with (out_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Optional: compare with Snort alerts (if used)
    snort_eids = parse_snort_alerts(SNORT_ALERT_FILE)
    if snort_eids:
        df["detected_by_snort"] = df["episode_id"].isin(snort_eids)
        snort_detection_rate = df["detected_by_snort"].mean()
        print(f"\nSnort-based detection rate: {snort_detection_rate:.2f}")
        df.to_csv(out_dir / "episodes_stats_with_snort.csv", index=False)


if __name__ == "__main__":
    main()
