# -*- coding: utf-8 -*-
"""The techniques the trained agent chose, as a graph for Neo4j.

How the techniques the agent decides on are used: how much each is used, what
follows what, and how much they overlap. This builds that graph from the
released corpus, over the 347 campaigns that reached the objective. The NEXT
graph, drawn with Neo4j Bloom, is prepared for print by v9_technique_figure.py.

  Technique nodes   campaigns that used it, total uses, repeats on itself
  NEXT              technique A was immediately followed by technique B
  WITH              A and B appear in the same campaign (the overlap), with
                    the Jaccard share of the campaigns that use either

Outputs, in results/technique_graph/:
  techniques.csv, next.csv, together.csv, load.cypher   for Neo4j Desktop
  load_inline.cypher   the same data written into the script, for pasting into
                       a query box (AuraDB) without uploading files

    python analysis/v8_technique_graph.py
"""
from __future__ import annotations

import csv
import itertools
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "results" / "telemetry" / "v5"
OUT = ROOT / "results" / "technique_graph"

ACRONYMS = {"sql": "SQL", "ssh": "SSH", "smb": "SMB", "rdp": "RDP", "c2": "C2",
            "lsass": "LSASS", "gpo": "GPO", "wmi": "WMI", "vpn": "VPN", "dc": "DC",
            "ntlm": "NTLM", "dll": "DLL"}


SPECIAL = {"AS_REP_ROASTING": "AS-REP roasting", "DCSYNC": "DCSync",
           "POWERSHELL_EXEC": "PowerShell execution", "PRIV_ESC_SUDO": "Privilege escalation (sudo)",
           "CREDS_IN_FILES": "Credentials in files", "DOMAIN_ACCT_DISCOVERY": "Domain account discovery",
           "C2_CHANNEL_ESTABLISH": "C2 channel", "CRED_DUMP_LSASS": "LSASS credential dump",
           "LATERAL_MOVE_SMB": "Lateral movement (SMB)", "RANSOMWARE_ENCRYPT": "Ransomware encryption",
           "BRUTE_FORCE_SSH": "SSH brute force"}


# Bloom wraps a node's text at about ten characters a line, over two lines, and
# breaks longer words mid-word ("Credential-s"). These short labels keep every
# word within ten characters and the whole label within two lines.
LABEL = {
    "C2_CHANNEL_ESTABLISH": "C2 channel", "DATA_FROM_LOCAL_SYSTEM": "Local data",
    "DISABLE_SECURITY_TOOLS": "Disable security", "LATERAL_MOVE_SMB": "Lateral (SMB)",
    "NETWORK_SCAN": "Network scan", "PRIV_ESC_SUDO": "Priv. esc. (sudo)",
    "SYSTEM_INFO_DISCOVERY": "System info", "ACCOUNT_MANIPULATION": "Account manip.",
    "CREDS_IN_FILES": "Creds in files", "ARCHIVE_COLLECTED_DATA": "Archive data",
    "SERVICE_STOP": "Service stop", "RDP_HIJACK": "RDP hijack", "CLEAR_LOGS": "Clear logs",
    "EXFILTRATE_DATA": "Exfiltrate data", "OBFUSCATED_FILES": "Obfuscate files",
    "WEB_SHELL_UPLOAD": "Web shell upload", "PHISHING_EMAIL": "Phishing email",
    "EXTERNAL_REMOTE_SERVICES": "External remote", "PASS_THE_TICKET": "Pass the ticket",
    "AS_REP_ROASTING": "AS-REP roasting", "SQL_INJECTION": "SQL injection",
    "KERBEROASTING": "Kerberoast", "DOMAIN_GROUP_DISCOVERY": "Domain groups",
    "NTLM_RELAY": "NTLM relay", "WMI_EXECUTION": "WMI execution",
    "DOMAIN_TRUST_DISCOVERY": "Domain trusts", "DOMAIN_ACCT_DISCOVERY": "Domain accounts",
    "INSTALL_BACKDOOR": "Install backdoor", "PASS_THE_HASH": "Pass the hash",
    # Bloom measures in pixels, so wide letters overflow sooner than the
    # character count suggests: "PowerShell" broke as "PowerShel-l".
    "POWERSHELL_EXEC": "Power- Shell", "REMOTE_SYSTEM_DISCOVERY": "Remote systems",
    "VALID_ACCOUNTS_LOGIN": "Valid accounts", "REGISTRY_RUN_KEYS": "Registry run keys",
    "PASSWORD_SPRAYING": "Password spraying", "BRUTE_FORCE_SSH": "SSH brute force",
    "DCSYNC": "DCSync", "CREDENTIALS_FROM_BROWSER": "Browser creds",
    "MODIFY_REGISTRY": "Modify registry", "CRED_DUMP_LSASS": "LSASS dump",
    "INHIBIT_SYSTEM_RECOVERY": "Inhibit recovery", "PROCESS_INJECTION": "Process injection",
    "RANSOMWARE_ENCRYPT": "Ransomware", "GPO_MODIFICATION": "GPO change",
    "GOLDEN_TICKET": "Golden ticket", "SCHEDULED_TASK": "Scheduled task",
}


# Official ATT&CK names for the identifiers the catalogue uses, for the key
# that goes with the drawings (the drawn labels are shortened to fit).
ATTACK_NAME = {
    "T1071.001": "Application Layer Protocol: Web Protocols", "T1005": "Data from Local System",
    "T1562.001": "Impair Defenses: Disable or Modify Tools", "T1021": "Remote Services",
    "T1046": "Network Service Discovery", "T1068": "Exploitation for Privilege Escalation",
    "T1082": "System Information Discovery", "T1098": "Account Manipulation",
    "T1552.001": "Unsecured Credentials: Credentials In Files",
    "T1560.001": "Archive Collected Data: Archive via Utility", "T1489": "Service Stop",
    "T1021.001": "Remote Services: Remote Desktop Protocol", "T1070": "Indicator Removal",
    "T1041": "Exfiltration Over C2 Channel", "T1027": "Obfuscated Files or Information",
    "T1505.003": "Server Software Component: Web Shell", "T1566": "Phishing",
    "T1133": "External Remote Services",
    "T1550.003": "Use Alternate Authentication Material: Pass the Ticket",
    "T1558.004": "Steal or Forge Kerberos Tickets: AS-REP Roasting",
    "T1190": "Exploit Public-Facing Application",
    "T1558.003": "Steal or Forge Kerberos Tickets: Kerberoasting",
    "T1069.002": "Permission Groups Discovery: Domain Groups",
    "T1557.001": "Adversary-in-the-Middle: LLMNR/NBT-NS Poisoning and SMB Relay",
    "T1047": "Windows Management Instrumentation", "T1482": "Domain Trust Discovery",
    "T1087.002": "Account Discovery: Domain Account", "T1543": "Create or Modify System Process",
    "T1550.002": "Use Alternate Authentication Material: Pass the Hash",
    "T1059.001": "Command and Scripting Interpreter: PowerShell",
    "T1547.001": "Boot or Logon Autostart Execution: Registry Run Keys / Startup Folder",
}


def fits_two_lines(label: str, width: int = 10) -> bool:
    lines, cur = [], ""
    for w in label.split():
        if len(w) > width:
            return False
        if cur and len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            if cur:
                lines.append(cur)
            cur = w
    lines.append(cur)
    return len(lines) <= 2


def caption(name: str) -> str:
    if name in SPECIAL:
        return SPECIAL[name]
    words = [ACRONYMS.get(w, w) for w in name.lower().split("_")]
    s = " ".join(words)
    return s[0].upper() + s[1:]


def main():
    meta = {}
    with open(CORPUS / "telemetry.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta.setdefault(r["technique"], (r["attack_id"], r["tactic"]))

    wins = []
    with open(CORPUS / "episodes.jsonl", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["objective_reached"]:
                wins.append([s.split("@")[0] for s in e["technique_sequence"]])

    campaigns, uses, repeats, nxt = Counter(), Counter(), Counter(), Counter()
    together = Counter()
    for seq in wins:
        uses.update(seq)
        present = sorted(set(seq))
        campaigns.update(present)
        for a, b in zip(seq, seq[1:]):
            if a == b:
                repeats[a] += 1
            else:
                nxt[(a, b)] += 1
        together.update(itertools.combinations(present, 2))

    n = len(wins)
    techs = sorted(meta, key=lambda t: (-campaigns[t], t))
    OUT.mkdir(parents=True, exist_ok=True)

    def band(c):
        share = c / n
        return "Core" if share >= 0.5 else "Common" if share >= 0.2 else "Occasional" if c else "Unused"

    nodes = [{"id": k, "name": t, "caption": caption(t), "attack_id": meta[t][0], "tactic": meta[t][1],
              "campaigns": campaigns[t], "share": round(100 * campaigns[t] / n, 1),
              "uses": uses[t], "repeats": repeats[t], "band": band(campaigns[t])}
             for k, t in enumerate(techs, 1)]
    ids = {d["name"]: d["id"] for d in nodes}
    edges_next = [{"from": a, "to": b, "count": c} for (a, b), c in nxt.most_common()]
    edges_with = [{"a": a, "b": b, "campaigns": c,
                   "jaccard": round(c / (campaigns[a] + campaigns[b] - c), 3)}
                  for (a, b), c in together.most_common()]

    for name, rows in (("techniques", nodes), ("next", edges_next), ("together", edges_with)):
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    (OUT / "load.cypher").write_text("""// Neo4j Desktop: copy the three CSV files into the database's import folder first.
CREATE CONSTRAINT technique_name IF NOT EXISTS FOR (t:Technique) REQUIRE t.name IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///techniques.csv' AS r
MERGE (t:Technique {name: r.name})
SET t.id = toInteger(r.id), t.caption = r.caption, t.attack_id = r.attack_id, t.tactic = r.tactic,
    t.campaigns = toInteger(r.campaigns), t.share = toFloat(r.share),
    t.uses = toInteger(r.uses), t.repeats = toInteger(r.repeats), t.band = r.band;

LOAD CSV WITH HEADERS FROM 'file:///next.csv' AS r
MATCH (a:Technique {name: r.from}), (b:Technique {name: r.to})
MERGE (a)-[e:NEXT]->(b) SET e.count = toInteger(r.count);

LOAD CSV WITH HEADERS FROM 'file:///together.csv' AS r
MATCH (a:Technique {name: r.a}), (b:Technique {name: r.b})
MERGE (a)-[e:WITH]->(b) SET e.campaigns = toInteger(r.campaigns), e.jaccard = toFloat(r.jaccard);
""", encoding="utf-8")

    # The inline version keeps only edges worth drawing, so it stays short
    # enough to paste: transitions seen at least 20 times, and pairs that share
    # at least half of the campaigns using either (Jaccard >= 0.5). Edges name
    # nodes by number rather than by technique name, for the same reason.
    keep_next = [e for e in edges_next if e["count"] >= 20]
    keep_with = [e for e in edges_with if e["jaccard"] >= 0.5]

    def lit(v):
        return json.dumps(v, ensure_ascii=False)

    node_rows = ",".join(
        "[" + ",".join(lit(x) for x in (d["id"], d["name"], d["caption"], d["attack_id"],
                                         d["tactic"], d["campaigns"], d["share"], d["uses"],
                                         d["repeats"], d["band"])) + "]" for d in nodes)
    next_rows = ",".join(f'[{ids[e["from"]]},{ids[e["to"]]},{e["count"]}]' for e in keep_next)
    with_rows = ",".join(f'[{ids[e["a"]]},{ids[e["b"]]},{e["campaigns"]},{e["jaccard"]}]'
                         for e in keep_with)
    statements = [
        "CREATE CONSTRAINT technique_id IF NOT EXISTS FOR (t:Technique) REQUIRE t.id IS UNIQUE",
        f"UNWIND [{node_rows}] AS r MERGE (t:Technique {{id: r[0]}}) "
        "SET t.name = r[1], t.caption = r[2], t.attack_id = r[3], t.tactic = r[4], "
        "t.campaigns = r[5], t.share = r[6], t.uses = r[7], t.repeats = r[8], t.band = r[9]",
        f"UNWIND [{next_rows}] AS r MATCH (a:Technique {{id: r[0]}}), (b:Technique {{id: r[1]}}) "
        "MERGE (a)-[e:NEXT]->(b) SET e.count = r[2]",
        f"UNWIND [{with_rows}] AS r MATCH (a:Technique {{id: r[0]}}), (b:Technique {{id: r[1]}}) "
        "MERGE (a)-[e:WITH]->(b) SET e.campaigns = r[2], e.jaccard = r[3]",
    ]
    assert set(LABEL) == set(meta), sorted(set(meta) ^ set(LABEL))
    too_long = [v for v in LABEL.values() if not fits_two_lines(v)]
    assert not too_long, too_long
    label_rows = ",".join(f"[{d['id']},{lit(LABEL[d['name']])}]" for d in nodes)
    statements.append(f"UNWIND [{label_rows}] AS r MATCH (t:Technique {{id: r[0]}}) SET t.label = r[1]")
    # Key for the drawings: every technique the flow view shows (it contains
    # the overlap view's techniques too). utf-8-sig so Excel reads Turkish.
    shown = {t for e in keep_next for t in (e["from"], e["to"])}
    shown_w = {t for e in keep_with for t in (e["a"], e["b"])}
    assert shown_w <= shown
    with open(OUT / "neo4j_exports" / "etiket_anahtari.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Şekildeki etiket", "ATT&CK no", "ATT&CK adı", "Taktik",
                    "Kullanan saldırı (347'de)", "Şekil 2'de"])
        for d in nodes:
            if d["name"] in shown:
                w.writerow([LABEL[d["name"]], d["attack_id"], ATTACK_NAME[d["attack_id"]], d["tactic"],
                            d["campaigns"], "evet" if d["name"] in shown_w else "hayır"])
    for k, s in enumerate(statements, 1):
        (OUT / f"inline_{k}.cypher").write_text(s + "\n", encoding="utf-8")
    (OUT / "load_inline.cypher").write_text(";\n\n".join(statements) + ";\n", encoding="utf-8")

    used = sum(1 for d in nodes if d["campaigns"])
    print(f"{n} campaigns, {used} of {len(nodes)} techniques used")
    print(f"NEXT: {len(edges_next)} distinct transitions, {len(keep_next)} seen >= 20 times")
    print(f"WITH: {len(edges_with)} pairs, {len(keep_with)} with Jaccard >= 0.5")
    print("bands:", Counter(d["band"] for d in nodes))
    print("statement sizes:", [len(s) for s in statements])
    print("top 10 by campaigns:")
    for d in nodes[:10]:
        print(f"  {d['caption']:32} {d['tactic']:22} {d['campaigns']:4} ({d['share']}%)  uses {d['uses']}")


if __name__ == "__main__":
    main()
