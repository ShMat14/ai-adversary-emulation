# -*- coding: utf-8 -*-
"""Generate the released ATT&CK-labelled telemetry corpus from a trained policy.

Why this artefact is different from the alternatives.

Microsoft Security (12 May 2026) state the problem plainly: "Gathering, labeling,
and maintaining datasets with real attack logs is costly and operationally
challenging. It requires not only labeling malicious activities, but also fully
reconstructing attack scenarios." Their answer generates log text with an LLM and
scores it with a second LLM against ground truth, which needs labelled data in
order to produce labelled data, and yields labels that are judged rather than
known.

Captured corpora -- Windows-APT 2025, ATLASv2 -- avoid that by running real
attacks in virtual machines. The telemetry is genuine but the corpus is fixed:
another scenario costs another run, and the labels are applied afterwards.

Here the label is not inferred at all. The environment emitted the event as part
of executing a specific technique, so the technique, its ATT&CK identifier, the
host, the platform and the detection rule that would fire are all known exactly
at emission time. And because a prerequisite mask gates every action, the trace
is executable by construction: an LLM can write a plausible log sequence for a
chain that could not have run in that order, and this cannot.

What is NOT claimed: these are constructed records, not captures from real hosts.
The corpus is not more realistic than a VM capture. It is exactly labelled,
auditable to a named rule, unbounded, and executable -- which is a different and
narrower claim.

    python analysis/v5_corpus.py --episodes 500 --out results/telemetry/v5
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.v5_train import make_env, load
from env.detection import PROFILES
from env.kill_chain_v5 import TECHNIQUES_V5


def generate(model_paths, topology="enterprise", episodes=500, max_steps=60,
             seed=1, start=None, explore=0.35):
    """Run the policies and collect every emitted event with full provenance.

    Two choices here are about the corpus rather than about the agent.

    Several checkpoints are used in rotation, and a fraction of episodes sample
    from the policy distribution instead of taking its argmax. A single
    deterministic policy converges on one preferred route and exercises about a
    quarter of the catalogue, which would make the released corpus a record of
    one operator's habits rather than of the technique set. Sampling stays inside
    the mask, so every trace remains executable by construction -- the property
    that distinguishes this artefact from generated log text.

    The exploratory episodes are marked in the per-episode metadata, so anyone
    training a detector on this can hold them out, weight them, or use them as
    the harder half.
    """
    import random as _random
    from sb3_contrib.common.maskable.utils import get_action_masks
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    models = [(os.path.basename(p), load("maskable", p)) for p in model_paths]
    env = make_env(topology, seed=seed, max_steps=max_steps)
    mm = env.model
    t0 = start or datetime(2026, 3, 2, 9, 14, tzinfo=timezone.utc)
    rows, episodes_meta = [], []
    clock = t0
    pick = _random.Random(20260905)

    for ep in range(episodes):
        model_name, model = models[ep % len(models)]
        stochastic = pick.random() < explore
        obs, _ = env.reset(seed=700_000 + ep)
        n0 = len(mm.detection.log)
        # what this episode's objective additionally required
        req = [k for k, v in (("persistence", mm.need_persist),
                              ("host_privilege", mm.need_hostpriv),
                              ("collection", mm.need_collect),
                              ("c2_channel", mm.need_c2)) if v]
        done, step = False, 0
        order = []
        while not done:
            a, _ = model.predict(obs, action_masks=get_action_masks(env),
                                 deterministic=not stochastic)
            tech, tgt = mm.decode(int(a))
            before = len(mm.detection.log)
            obs, r, term, trunc, info = env.step(int(a))
            step += 1
            done = term or trunc
            for ev in mm.detection.log[before:]:
                clock += timedelta(seconds=7 + (len(rows) % 23))
                prof = PROFILES[ev.technique]
                rows.append({
                    "timestamp": clock.isoformat(),
                    "episode": ep,
                    "step": step,
                    "host": ev.host,
                    "platform": mm.topo.os_of(ev.host),
                    "subnet": mm.topo.subnet_of[ev.host],
                    "channel": ev.channel,
                    "event_id": ev.event_id,
                    "event_name": ev.name,
                    "technique": ev.technique,
                    "attack_id": ev.mitre_id,
                    "tactic": TECHNIQUES_V5[ev.technique].tactic,
                    "suspicious": ev.suspicious,
                    "detection_rule": prof.rule,
                    "action_advanced_state": bool(info.get("advanced")),
                    "label": "malicious",
                })
            order.append(f"{info['technique']}@{info['target']}")
        episodes_meta.append({
            "episode": ep, "steps": step,
            "policy": model_name,
            "action_selection": "sampled" if stochastic else "greedy",
            # what this engagement offered. A detector trained on the corpus
            # should be able to see that a chain used AS-REP roasting because
            # Kerberoasting was not on the table, rather than by preference.
            "techniques_withdrawn": sorted(
                set(TECHNIQUES_V5) - mm.available),
            "objective_reached": bool(mm.is_goal()),
            "incident_declared": bool(mm.caught()),
            "extra_requirements": req,
            "events": len(mm.detection.log) - n0,
            "technique_sequence": order,
        })
    return rows, episodes_meta


def write(rows, meta, outdir, topology, max_steps):
    os.makedirs(outdir, exist_ok=True)
    jsonl = os.path.join(outdir, "telemetry.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    csv_path = os.path.join(outdir, "telemetry.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ep_path = os.path.join(outdir, "episodes.jsonl")
    with open(ep_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    digest = hashlib.sha256(open(jsonl, "rb").read()).hexdigest()
    techs = sorted({r["technique"] for r in rows})
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "generator": "analysis/v5_corpus.py",
        "policies": sorted({m["policy"] for m in meta}),
        "exploratory_episodes": sum(
            1 for m in meta if m["action_selection"] == "sampled"),
        "topology": topology,
        "episode_step_budget": max_steps,
        "episodes": len(meta),
        "events": len(rows),
        "distinct_techniques": len(techs),
        "distinct_attack_ids": len({r["attack_id"] for r in rows}),
        "channels": sorted({r["channel"] for r in rows}),
        "objective_reached": sum(m["objective_reached"] for m in meta),
        "incidents_declared": sum(m["incident_declared"] for m in meta),
        "techniques": techs,
        "catalogue_size": len(TECHNIQUES_V5),
        "catalogue_coverage": round(100.0 * len(techs) / len(TECHNIQUES_V5), 1),
        "records_per_technique": {
            k: sum(1 for r in rows if r["technique"] == k) for k in techs},
        # Named rather than left as a silent gap. A technique absent from the
        # corpus is one the policies did not choose, not one the environment
        # cannot produce; they are alternatives that a competent policy passes
        # over in favour of a cheaper or quieter route to the same outcome.
        "techniques_absent": sorted(set(TECHNIQUES_V5) - set(techs)),
        "sha256_telemetry_jsonl": digest,
        "python": platform.python_version(),
        "labelling": ("exact: each record is emitted by the technique that "
                      "produced it, so the ATT&CK identifier, host, platform and "
                      "candidate detection rule are known at emission time and "
                      "are not inferred, judged or applied retrospectively"),
        "caveat": ("records are constructed by a simulation, not captured from "
                   "real hosts; the corpus is exactly labelled and executable, "
                   "not more realistic than a VM capture"),
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest, [jsonl, csv_path, ep_path]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["results/models/v5/masked_enterprise_s0"],
                    help="one or more checkpoints, used in rotation")
    ap.add_argument("--explore", type=float, default=0.35,
                    help="fraction of episodes that sample instead of taking "
                         "the argmax; masked either way, so still executable")
    ap.add_argument("--topology", default="enterprise")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--out", default="results/telemetry/v5")
    a = ap.parse_args()

    rows, meta = generate(a.models, a.topology, a.episodes, a.max_steps,
                          explore=a.explore)
    man, files = write(rows, meta, a.out, a.topology, a.max_steps)
    print(f"  episodes            {man['episodes']}")
    print(f"  events              {man['events']:,}")
    print(f"  distinct techniques {man['distinct_techniques']}")
    print(f"  distinct ATT&CK ids {man['distinct_attack_ids']}")
    print(f"  catalogue coverage  {man['catalogue_coverage']}% "
          f"({man['distinct_techniques']} of {man['catalogue_size']})")
    print(f"  policies            {', '.join(man['policies'])}")
    print(f"  exploratory eps     {man['exploratory_episodes']}/{man['episodes']}")
    if man["techniques_absent"]:
        print(f"  never chosen        {', '.join(man['techniques_absent'])}")
    print(f"  channels            {', '.join(man['channels'])}")
    print(f"  objective reached   {man['objective_reached']}/{man['episodes']}")
    print(f"  incidents declared  {man['incidents_declared']}/{man['episodes']}")
    for f in files:
        print(f"  wrote {f}  ({os.path.getsize(f)/1024:.0f} KB)")
