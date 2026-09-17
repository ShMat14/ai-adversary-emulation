# -*- coding: utf-8 -*-
"""
Check every identifier in analysis/attack_catalogue.py against MITRE's official
ATT&CK Enterprise bundle.

The catalogue was curated by hand. ATT&CK renames techniques, promotes
sub-techniques and deprecates entries between versions, so the identifiers must
be verified against the published data before any of them appear in the paper.

    # once, ~40 MB, from the official MITRE repository:
    curl -L -o data/enterprise-attack.json \
      https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json

    python analysis/validate_catalogue.py
    python analysis/validate_catalogue.py --bundle path/to/enterprise-attack.json
    python analysis/validate_catalogue.py --fix-names     # rewrite names to match MITRE

Exit code is 0 only when every identifier resolves to a current, non-deprecated
technique whose name matches. Anything else is a finding to resolve by hand.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.attack_catalogue import INAPPLICABLE, IMPLEMENTED_IDS

DEFAULT_BUNDLE = os.path.join("data", "enterprise-attack.json")
SOURCE = ("https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
          "master/enterprise-attack/enterprise-attack.json")

# platforms the modelled network actually presents. A structural entry that
# ATT&CK says runs on Windows is a curation error and must be re-tiered.
MODELLED_PLATFORMS = {"Windows"}


def load_bundle(path):
    if not os.path.exists(path):
        sys.exit(
            f"ATT&CK bundle not found at {path}\n"
            f"Download it once from:\n  {SOURCE}\n"
            f"or pass --bundle with the path to a local copy."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    techniques = {}
    for obj in raw.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        ext = next((r for r in obj.get("external_references", [])
                    if r.get("source_name") == "mitre-attack"), None)
        if not ext or "external_id" not in ext:
            continue
        techniques[ext["external_id"]] = {
            "name": obj.get("name", ""),
            "deprecated": bool(obj.get("x_mitre_deprecated")
                               or obj.get("revoked")),
            "platforms": set(obj.get("x_mitre_platforms", [])),
        }
    return techniques


def short_name(full):
    """`Weaken Encryption: Reduce Key Space` -> `Reduce Key Space`.

    ATT&CK stores a sub-technique's own name without the parent prefix; the
    catalogue spells out both for readability, so compare on the tail.
    """
    return full.split(":", 1)[1].strip() if ":" in full else full.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--fix-names", action="store_true",
                    help="print a corrected catalogue block to stdout")
    args = ap.parse_args()

    mitre = load_bundle(args.bundle)
    print(f"loaded {len(mitre)} ATT&CK Enterprise techniques from {args.bundle}")

    missing, deprecated, renamed, mistiered = [], [], [], []

    for t in INAPPLICABLE:
        ref = mitre.get(t.id)
        if ref is None:
            missing.append(t)
            continue
        if ref["deprecated"]:
            deprecated.append((t, ref))
        if short_name(t.name).lower() != short_name(ref["name"]).lower():
            renamed.append((t, ref))
        # a "structural" entry that ATT&CK lists for Windows is mis-tiered:
        # the asset does exist here, so the claim would be wrong
        if t.tier == "structural" and ref["platforms"] & MODELLED_PLATFORMS:
            mistiered.append((t, ref))

    for t in sorted(IMPLEMENTED_IDS):
        if t and t not in mitre:
            missing.append(type("X", (), {"id": t, "name": "(implemented action)"})())

    def report(title, rows, fmt):
        if rows:
            print(f"\n{title} ({len(rows)}):")
            for row in rows:
                print("  " + fmt(row))

    report("NOT FOUND in ATT&CK", missing, lambda t: f"{t.id}  {t.name}")
    report("DEPRECATED or REVOKED", deprecated,
           lambda r: f"{r[0].id}  {r[0].name}")
    report("NAME MISMATCH", renamed,
           lambda r: f"{r[0].id}  catalogue={short_name(r[0].name)!r}  "
                     f"mitre={r[1]['name']!r}")
    report("MIS-TIERED (structural, but ATT&CK lists Windows)", mistiered,
           lambda r: f"{r[0].id}  {r[0].name}  platforms={sorted(r[1]['platforms'])}")

    if args.fix_names:
        print("\n--- corrected names ---")
        for t, ref in renamed:
            prefix = t.name.split(":", 1)[0] + ": " if ":" in t.name else ""
            print(f'    ("{t.id}", "{prefix}{ref["name"]}"),')

    bad = len(missing) + len(deprecated) + len(renamed) + len(mistiered)
    if bad == 0:
        print(f"\nOK: all {len(INAPPLICABLE)} catalogue identifiers verified "
              f"against MITRE.")
        return 0
    print(f"\n{bad} finding(s) to resolve before publication.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
