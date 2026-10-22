# -*- coding: utf-8 -*-
"""Do the Windows events the environment emits appear when the technique really runs?

The corpus is constructed: each technique emits the records it would leave,
each named by channel, event identifier and a description. This checks them
against OTRF Security-Datasets (github.com/OTRF/Security-Datasets, MIT
licence), a public collection of Windows host logs recorded while ATT&CK
techniques ran in a lab domain. Only the Windows channels can be compared
(Security, Sysmon, PowerShell, System, TerminalServices, WMI-Activity); the
auditd, syslog and WAF records have no counterpart there.

Matching is strict. A recording tests a technique only when its metadata names
exactly the technique's ATT&CK identifier (a recording of T1021.006, WinRM, is
not a test of our T1021, which moves over SMB), only its host logs are read,
and a recording with no Windows events at all (a Linux capture filed under the
same identifier) is set aside.

An identifier such as logon 4624 or Sysmon 10 occurs in almost any Windows
log, so presence alone shows little. Where the event's own description in the
catalogue states a condition, the condition is applied: "Network logon type 3"
requires LogonType 3, "handle to lsass.exe" requires lsass.exe as the target,
"(RC4)" requires encryption type 0x17, and so on (CONDITIONS below). The
conditions are read from the descriptions, not chosen to fit the data. Each
watched event is then counted in the technique's recordings and its rate there
compared with its rate in the other recordings.

The archives are downloaded once into --data (outside the repository) and read
as data; nothing in them is executed. Writes results/v9_otrf_check.json.

    python -I analysis/v9_otrf_check.py --data <folder> [--download]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results" / "telemetry" / "v5" / "telemetry.csv"
OUT = ROOT / "results" / "v9_otrf_check.json"
REPO = "OTRF/Security-Datasets"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/"
WINDOWS = {"Security", "Sysmon", "PowerShell", "System", "TerminalServices", "WMI-Activity"}


def _s(f, k):
    return str(f.get(k, "") or "").lower()


def _proc(f):
    return _s(f, "NewProcessName").replace("/", "\\").rsplit("\\", 1)[-1]


def _hex(v):
    try:
        return int(str(v), 16)
    except ValueError:
        return 0


# Description in the catalogue -> the condition that description states.
CONDITIONS = {
    "ProcessAccess: handle to lsass.exe": lambda f: _s(f, "TargetImage").endswith("lsass.exe"),
    "Directory replication (DS-Replication-Get-Changes)":
        lambda f: any(g in _s(f, "Properties") for g in ("1131f6aa-9c07-11d1-f79f-00c04fc2dcd2",
                                                           "1131f6ad-9c07-11d1-f79f-00c04fc2dcd2")),
    "Network logon type 3": lambda f: _s(f, "LogonType") == "3",
    "Network logon type 3 with NTLM":
        lambda f: _s(f, "LogonType") == "3" and "ntlm" in _s(f, "AuthenticationPackageName"),
    "Network share accessed (ADMIN$)": lambda f: "admin$" in _s(f, "ShareName"),
    "Process Create: child of WmiPrvSE.exe": lambda f: _s(f, "ParentProcessName").endswith("wmiprvse.exe"),
    "ProcessAccess with PROCESS_VM_WRITE": lambda f: bool(_hex(f.get("GrantedAccess", "0")) & 0x20),
    "Process Create: net group /domain":
        lambda f: _proc(f) in ("net.exe", "net1.exe") and "group" in _s(f, "CommandLine"),
    "Process Create: net view / nltest":
        lambda f: (_proc(f) in ("net.exe", "net1.exe") and "view" in _s(f, "CommandLine"))
        or _proc(f) == "nltest.exe",
    "RegistryEvent: value set under a Run key": lambda f: "\\currentversion\\run" in _s(f, "TargetObject"),
    "Kerberos service ticket (RC4) requested": lambda f: _s(f, "TicketEncryptionType") in ("0x17", "23"),
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "v9_otrf_check"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def channel(name: str) -> str:
    if name in ("Security", "System"):
        return name
    for key, ours in (("Sysmon", "Sysmon"), ("PowerShell", "PowerShell"),
                      ("WMI-Activity", "WMI-Activity"), ("TerminalServices", "TerminalServices")):
        if key in name:
            return ours
    return "other"


def metadata():
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/master?recursive=1"))
    paths = [i["path"] for i in tree["tree"]
             if i["path"].startswith("datasets/atomic/_metadata/") and i["path"].endswith(".yaml")]
    recs = []
    for p in sorted(paths):
        txt = fetch(RAW + p).decode("utf-8", "replace")
        ids = set()
        for m in re.finditer(r"technique:\s*['\"]?(T\d{4})['\"]?\s*\n\s*sub-technique:\s*['\"]?(\d{3})?", txt):
            ids.add(m.group(1) + (f".{m.group(2)}" if m.group(2) else ""))
        # Host logs only: a few metadata files label a network capture "Host".
        host = [u for u in re.findall(r"type:\s*Host\s*\n\s*link:\s*(\S+)", txt) if "/host/" in u]
        title = (re.search(r"^title:\s*(.+)$", txt, re.M) or [None, ""])[1].strip()
        recs.append({"id": p.rsplit("/", 1)[-1][:-5], "title": title, "attack_ids": sorted(ids),
                     "host_files": host})
    return recs


def our_events():
    """Windows events per ATT&CK identifier: {(channel, id, description): watched}."""
    events, name = defaultdict(dict), {}
    with CORPUS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["channel"] not in WINDOWS:
                continue
            key = (r["channel"], int(r["event_id"]), r["event_name"])
            events[r["attack_id"]][key] = events[r["attack_id"]].get(key, False) or r["suspicious"] == "True"
            name[r["attack_id"]] = r["technique"]
    return events, name


def read_events(path: Path, wanted) -> Counter:
    """Counts of ('id', channel, event) and of ('cond', description) in one archive."""
    counts = Counter()
    by_id = defaultdict(list)
    for ch, eid, desc in wanted:
        if desc in CONDITIONS:
            by_id[(ch, eid)].append(desc)

    def consume(stream):
        for line in io.TextIOWrapper(stream, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            # Two layouts: flat (Channel, EventID) and the 2019 Winlogbeat one
            # (log_name, event_id, fields under event_data).
            ch = e.get("Channel", e.get("log_name"))
            eid = e.get("EventID", e.get("event_id"))
            if ch is None or eid is None:
                continue
            try:
                key = (channel(str(ch)), int(eid))
            except (TypeError, ValueError):
                continue
            counts[("id",) + key] += 1
            if key in by_id:
                fields = {**e, **(e.get("event_data") or {})}
                for desc in by_id[key]:
                    if CONDITIONS[desc](fields):
                        counts[("cond", desc)] += 1

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".json"):
                    with z.open(n) as s:
                        consume(s)
    else:
        with tarfile.open(path) as t:
            for m in t.getmembers():
                if m.isfile() and m.name.endswith(".json"):
                    consume(t.extractfile(m))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder for the archives, outside the repository")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)

    events, name = our_events()
    recs = metadata()
    matched = {t: [r for r in recs if t in r["attack_ids"] and r["host_files"]] for t in events}
    matched = {t: rs for t, rs in matched.items() if rs}
    files = sorted({f for rs in matched.values() for r in rs for f in r["host_files"]})
    local = {f: data / f.split("/datasets/atomic/")[1].replace("/", "__") for f in files}
    if a.download:
        for f, p in local.items():
            if not p.exists():
                p.write_bytes(fetch(f))
    missing = [f for f, p in local.items() if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} archives not downloaded; rerun with --download")

    wanted = {k for ev in events.values() for k in ev}
    counts = {f: read_events(p, wanted) for f, p in local.items()}
    windows_events = {f: sum(v for k, v in c.items() if k[0] == "id" and k[1] in WINDOWS)
                      for f, c in counts.items()}
    set_aside = sorted(f for f, n in windows_events.items() if n == 0)
    usable = [f for f in files if f not in set_aside]

    def n_of(c, key):
        ch, eid, desc = key
        return c[("cond", desc)] if desc in CONDITIONS else c[("id", ch, eid)]

    rows = []
    for t in sorted(matched):
        mine = [f for r in matched[t] for f in r["host_files"] if f in usable]
        if not mine:
            continue
        others = [f for f in usable if f not in mine]
        own, rest = Counter(), Counter()
        for f in mine:
            own.update(counts[f])
        for f in others:
            rest.update(counts[f])
        n_own = sum(windows_events[f] for f in mine)
        n_rest = sum(windows_events[f] for f in others)
        checks = []
        for key, watched in sorted(events[t].items()):
            k_own, k_rest = n_of(own, key), n_of(rest, key)
            ratio = (k_own / n_own) / ((k_rest + 0.5) / n_rest)
            # Why an event is missing, where the data can say: the identifier
            # never logged anywhere in the collection, or its channel absent
            # from this technique's recordings. Otherwise it is simply absent.
            id_anywhere = sum(counts[f][("id", key[0], key[1])] for f in usable)
            channel_here = sum(v for k, v in own.items() if k[0] == "id" and k[1] == key[0])
            reading = ("present, 2x or more" if k_own and ratio >= 2 else "present" if k_own
                       else "identifier never logged in the collection" if not id_anywhere
                       else "channel not in the recording" if not channel_here else "absent")
            checks.append({"channel": key[0], "event_id": key[1], "description": key[2],
                           "condition_applied": key[2] in CONDITIONS, "watched": watched,
                           "in_recordings": k_own, "in_other_recordings": k_rest,
                           "rate_ratio": round(ratio, 1), "reading": reading})
        used = [r for r in matched[t] if any(f in usable for f in r["host_files"])]
        rows.append({"attack_id": t, "technique": name[t],
                     "recordings": [r["id"] for r in used], "titles": [r["title"] for r in used],
                     "windows_events": n_own, "checks": checks})

    w = [c for r in rows for c in r["checks"] if c["watched"]]
    summary = {
        "techniques_with_windows_events": len(events),
        "techniques_with_exact_recording": len(rows),
        "host_archives_read": len(usable),
        "archives_set_aside_no_windows_events": [f.split("/datasets/atomic/")[1] for f in set_aside],
        "watched_events": len(w),
        "watched_events_with_condition": sum(c["condition_applied"] for c in w),
        "watched_events_present": sum(c["in_recordings"] > 0 for c in w),
        "watched_events_at_least_2x": sum(c["in_recordings"] > 0 and c["rate_ratio"] >= 2 for c in w),
        "techniques_all_watched_present": sum(all(c["in_recordings"] > 0 for c in r["checks"] if c["watched"])
                                              for r in rows),
    }
    OUT.write_text(json.dumps({"summary": summary, "techniques": rows,
                               "no_exact_recording": sorted(set(events) - {r["attack_id"] for r in rows})},
                              indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(f"\n{r['attack_id']} {r['technique']} ({r['windows_events']:,} Windows events, "
              f"{len(r['recordings'])} recording(s))")
        for c in r["checks"]:
            print(f"   {'W' if c['watched'] else ' '}{'c' if c['condition_applied'] else ' '} "
                  f"{c['channel']:13} {c['event_id']:>5} {c['description'][:44]:44} "
                  f"here {c['in_recordings']:>5}  elsewhere {c['in_other_recordings']:>6}  ratio {c['rate_ratio']}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
