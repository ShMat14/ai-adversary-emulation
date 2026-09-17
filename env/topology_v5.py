# -*- coding: utf-8 -*-
"""Network topologies for the v5 environment.

v4 hard-coded three hosts and a scripted progression user01 -> srv01 -> dc01.
The agent chose a technique; the environment chose the target. That makes the
problem materially easier than the formulation the field uses, where an action
is a (host, technique) pair and the agent must choose where to act as well as
what to do -- see Koo et al. (ETRI Journal, 2026), which states the action space
as the Cartesian product A = H x T.

This module supplies the network as data so topology size is a variable rather
than a constant. `v4compat` reproduces the original three-host chain, so the
change in action-space formulation can be measured with the network held fixed;
`enterprise` is a six-subnet estate comparable in scale to the largest scenario
in that paper.
"""

# A subnet is reachable from another if an edge exists. The attacker begins
# outside, so only subnets adjacent to 0 are initially exposed.
TOPOLOGIES = {
    # ---- the original v4 network, for a controlled comparison -------------
    # One exposed Linux web host: the shape `Target/vulnerable_app.py` actually
    # has. Used only by the sim-to-real arm of Section 4.8, so that the policy
    # evaluated in simulation and the policy executed against the live target
    # face the same observation and the same action space.
    "webhost": {
        "subnets": {"1": ["web01"]},
        "edges": [[0, 1]],
        "services": {"web01": ["web", "ssh"]},
        "os": {"web01": "linux"},
        "goal": "web01",
    },
    "v4compat": {
        "subnets": {1: ["user01"], 2: ["srv01"], 3: ["dc01"]},
        "edges": [(0, 1), (1, 2), (2, 3)],
        "services": {"user01": ["smb", "web"], "srv01": ["smb", "ssh"], "dc01": ["smb", "ldap"]},
        "os": {"user01": "windows", "srv01": "windows", "dc01": "windows"},
        "goal": "dc01",
    },
    # ---- estates of other sizes, for the transfer-across-scale test -------
    # Tiered like "enterprise" so that size is the variable and shape is held
    # roughly constant. Usable only in the size-invariant mode of the
    # environment, since the observation width otherwise follows the host count.
    "small8": {
        "subnets": {1: ["web01", "mail01"],
                    2: ["user01", "user02"],
                    3: ["srv01", "file01"],
                    4: ["dc01", "db01"]},
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "services": {
            "web01": ["web", "ssh", "vpn"], "mail01": ["web", "smtp"],
            "user01": ["smb"], "user02": ["smb", "rdp"],
            "srv01": ["smb", "ssh"], "file01": ["smb"],
            "dc01": ["smb", "ldap"], "db01": ["smb", "sql"],
        },
        "os": {"web01": "linux", "mail01": "linux",
               "user01": "windows", "user02": "windows",
               "srv01": "windows", "file01": "linux",
               "dc01": "windows", "db01": "linux"},
        "goal": "dc01",
    },
    "mid16": {
        "subnets": {1: ["web01", "mail01", "proxy01"],
                    2: ["user01", "user02", "user03", "user04"],
                    3: ["srv01", "srv02", "file01", "app01"],
                    4: ["admin01", "jump01", "build01"],
                    5: ["dc01", "db01"]},
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 4)],
        "services": {
            "web01": ["web", "ssh", "vpn"], "mail01": ["web", "smtp"],
            "proxy01": ["web", "ssh"],
            "user01": ["smb"], "user02": ["smb"], "user03": ["smb", "rdp"],
            "user04": ["smb"],
            "srv01": ["smb", "ssh"], "srv02": ["smb", "rdp"],
            "file01": ["smb"], "app01": ["smb", "web"],
            "admin01": ["smb", "rdp"], "jump01": ["ssh", "rdp"],
            "build01": ["smb", "ssh"],
            "dc01": ["smb", "ldap"], "db01": ["smb", "sql"],
        },
        "os": {"web01": "linux", "mail01": "linux", "proxy01": "linux",
               "user01": "windows", "user02": "windows", "user03": "windows",
               "user04": "windows",
               "srv01": "windows", "srv02": "windows", "file01": "linux",
               "app01": "linux", "admin01": "windows", "jump01": "linux",
               "build01": "linux", "dc01": "windows", "db01": "linux"},
        "goal": "dc01",
    },
    "large20": {
        "subnets": {1: ["web01", "mail01", "proxy01"],
                    2: ["user01", "user02", "user03", "user04", "user05"],
                    3: ["srv01", "srv02", "srv03", "file01", "app01"],
                    4: ["admin01", "jump01", "build01", "mon01"],
                    5: ["dc01", "dc02", "db01"]},
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 4)],
        "services": {
            "web01": ["web", "ssh", "vpn"], "mail01": ["web", "smtp"],
            "proxy01": ["web", "ssh"],
            "user01": ["smb"], "user02": ["smb"], "user03": ["smb", "rdp"],
            "user04": ["smb"], "user05": ["smb", "rdp"],
            "srv01": ["smb", "ssh"], "srv02": ["smb", "rdp"],
            "srv03": ["smb", "web"], "file01": ["smb"], "app01": ["smb", "web"],
            "admin01": ["smb", "rdp"], "jump01": ["ssh", "rdp"],
            "build01": ["smb", "ssh"], "mon01": ["web", "ssh"],
            "dc01": ["smb", "ldap"], "dc02": ["smb", "ldap"],
            "db01": ["smb", "sql"],
        },
        "os": {"web01": "linux", "mail01": "linux", "proxy01": "linux",
               "user01": "windows", "user02": "windows", "user03": "windows",
               "user04": "windows", "user05": "windows",
               "srv01": "windows", "srv02": "windows", "srv03": "windows",
               "file01": "linux", "app01": "linux",
               "admin01": "windows", "jump01": "linux", "build01": "linux",
               "mon01": "linux",
               "dc01": "windows", "dc02": "windows", "db01": "linux"},
        "goal": "dc01",
    },
    # ---- three further estates for the zero-shot transfer test ------------
    # Each holds exactly twelve hosts so that a policy trained on "enterprise"
    # can be evaluated on them unchanged; what differs is the shape.
    #
    # flat12: two zones, so once past the DMZ nearly everything is adjacent.
    "flat12": {
        "subnets": {1: ["web01", "mail01"],
                    2: ["user01", "user02", "user03", "srv01", "srv02",
                        "file01", "admin01", "jump01", "dc01", "db01"]},
        "edges": [(0, 1), (1, 2)],
        "services": {
            "web01": ["web", "ssh", "vpn"], "mail01": ["web", "smtp"],
            "user01": ["smb"], "user02": ["smb"], "user03": ["smb", "rdp"],
            "srv01": ["smb", "ssh"], "srv02": ["smb", "rdp"], "file01": ["smb"],
            "admin01": ["smb", "rdp"], "jump01": ["ssh", "rdp"],
            "dc01": ["smb", "ldap"], "db01": ["smb", "sql"],
        },
        "os": {"web01": "linux", "mail01": "linux",
               "user01": "windows", "user02": "windows", "user03": "windows",
               "srv01": "windows", "srv02": "windows", "file01": "linux",
               "admin01": "windows", "jump01": "linux",
               "dc01": "windows", "db01": "linux"},
        "goal": "dc01",
    },
    # hub12: every internal zone hangs off a single jump host.
    "hub12": {
        "subnets": {1: ["web01", "mail01"],
                    2: ["jump01"],
                    3: ["user01", "user02", "user03"],
                    4: ["srv01", "srv02", "file01"],
                    5: ["admin01", "dc01", "db01"]},
        "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5)],
        "services": {
            "web01": ["web", "ssh", "vpn"], "mail01": ["web", "smtp"],
            "jump01": ["ssh", "rdp", "smb"],
            "user01": ["smb"], "user02": ["smb"], "user03": ["smb", "rdp"],
            "srv01": ["smb", "ssh"], "srv02": ["smb", "rdp"], "file01": ["smb"],
            "admin01": ["smb", "rdp"], "dc01": ["smb", "ldap"],
            "db01": ["smb", "sql"],
        },
        "os": {"web01": "linux", "mail01": "linux", "jump01": "linux",
               "user01": "windows", "user02": "windows", "user03": "windows",
               "srv01": "windows", "srv02": "windows", "file01": "linux",
               "admin01": "windows", "dc01": "windows", "db01": "linux"},
        "goal": "dc01",
    },
    # deep12: six zones of two hosts in a line. Maximum depth.
    "deep12": {
        "subnets": {1: ["web01", "mail01"],
                    2: ["user01", "user02"],
                    3: ["srv01", "srv02"],
                    4: ["file01", "app01"],
                    5: ["admin01", "jump01"],
                    6: ["dc01", "db01"]},
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        "services": {
            "web01": ["web", "ssh", "vpn"], "mail01": ["web", "smtp"],
            "user01": ["smb"], "user02": ["smb", "rdp"],
            "srv01": ["smb", "ssh"], "srv02": ["smb", "rdp"],
            "file01": ["smb"], "app01": ["smb", "web"],
            "admin01": ["smb", "rdp"], "jump01": ["ssh", "rdp"],
            "dc01": ["smb", "ldap"], "db01": ["smb", "sql"],
        },
        "os": {"web01": "linux", "mail01": "linux",
               "user01": "windows", "user02": "windows",
               "srv01": "windows", "srv02": "windows",
               "file01": "linux", "app01": "linux",
               "admin01": "windows", "jump01": "linux",
               "dc01": "windows", "db01": "linux"},
        "goal": "dc01",
    },
    # ---- a six-subnet enterprise estate -----------------------------------
    # DMZ -> workstations -> servers -> admin tier -> domain core. Lateral
    # movement is a genuine choice: several hosts sit at each tier, and only
    # some carry the credentials or the service a given technique needs.
    "enterprise": {
        "subnets": {
            1: ["web01", "mail01"],                 # DMZ, internet-facing
            2: ["user01", "user02", "user03"],      # workstations
            3: ["srv01", "srv02", "file01"],        # application/file servers
            4: ["admin01", "jump01"],               # administrative tier
            5: ["dc01", "db01"],                    # domain core
        },
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 4)],
        "services": {
            # web01 also fronts the remote-access concentrator, which is what
            # T1133 (External Remote Services) and VPN password spraying abuse.
            # Without a "vpn" service anywhere on the exposed edge, both of those
            # techniques were in the catalogue and never once legal.
            "web01": ["web", "ssh", "vpn"], "mail01": ["web", "smtp"],
            "user01": ["smb"], "user02": ["smb"], "user03": ["smb", "rdp"],
            "srv01": ["smb", "ssh"], "srv02": ["smb", "rdp"], "file01": ["smb"],
            "admin01": ["smb", "rdp"], "jump01": ["ssh", "rdp"],
            "dc01": ["smb", "ldap"], "db01": ["smb", "sql"],
        },
        # A real estate is mixed. This matters here more than it does for a
        # pure simulator: our detection model emits real Windows Security and
        # Sysmon event identifiers, and those events do not exist on Linux. An
        # emitted corpus that logged Event 4624 on a Linux host would not
        # survive inspection by the defenders it is meant to serve.
        "os": {
            "web01": "linux", "mail01": "linux",
            "user01": "windows", "user02": "windows", "user03": "windows",
            "srv01": "windows", "srv02": "windows", "file01": "linux",
            "admin01": "windows", "jump01": "linux",
            "dc01": "windows", "db01": "linux",
        },
        "goal": "dc01",
    },
}


class Topology:
    """Hosts, subnets and reachability, derived from one of the presets above."""

    def __init__(self, name="enterprise"):
        if name not in TOPOLOGIES:
            raise ValueError(f"unknown topology {name!r}; have {sorted(TOPOLOGIES)}")
        spec = TOPOLOGIES[name]
        self.name = name
        self.subnets = {int(k): list(v) for k, v in spec["subnets"].items()}
        self.services = dict(spec["services"])
        self.os = dict(spec.get("os", {}))
        self.goal = spec["goal"]
        self.hosts = [h for sub in self.subnets.values() for h in sub]
        self.subnet_of = {h: s for s, hs in self.subnets.items() for h in hs}
        self._adj = {}
        for a, b in spec["edges"]:
            self._adj.setdefault(a, set()).add(b)
            self._adj.setdefault(b, set()).add(a)

    def adjacent(self, s1, s2):
        return s2 in self._adj.get(s1, ())

    def exposed(self, host):
        """Reachable from outside the estate (subnet 0) with no foothold."""
        return self.adjacent(0, self.subnet_of[host])

    def reachable_from(self, src, dst):
        """Can a foothold on `src` reach `dst`? Same subnet, or an adjacent one."""
        a, b = self.subnet_of[src], self.subnet_of[dst]
        return a == b or self.adjacent(a, b)

    def has_service(self, host, svc):
        return svc in self.services.get(host, ())

    def os_of(self, host):
        return self.os.get(host, "windows")

    def is_windows(self, host):
        return self.os_of(host) == "windows"

    def __len__(self):
        return len(self.hosts)
