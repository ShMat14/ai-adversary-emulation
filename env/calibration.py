# -*- coding: utf-8 -*-
"""Measured success rates, and a sensitivity control built from them.

Fifteen techniques in the catalogue can be executed for real against a web target
(`Target/vulnerable_app.py`), which shares no code and no state with the
simulator. `analysis/v5_calibration.py` measures what an executing agent
actually achieves. Every one of the sixteen measurements came back *lower* than the
simulator assumes, by 0.230 on average.

That is a finding about our modelling, and it raises an obvious question about
every result in the paper: if the environment is systematically optimistic, does
the headline survive a pessimistic one?

This module answers it two ways.

  MEASURED   the measured rates, substituted directly.
  scale      a uniform multiplier applied to all forty-five, so the whole
             catalogue can be made as pessimistic as the measurement suggests
             and the results recomputed.

The scale factor worth reporting is the mean ratio of observed to assumed across
the sixteen measurements, which is what a reader would apply if they believed our
measurement generalised to the thirty techniques we cannot execute.

Nothing here is applied by default. The environment ships with its documented
probabilities, and calibration is opt-in so that a result computed under it is
never confused with one computed under the shipped values.
"""

# analysis/v5_calibration.py, 300 real executions per row, seed 2026.
# Where a technique has two variants the harder one is taken.
MEASURED = {
    "ACCOUNT_MANIPULATION": 0.50,             # assumed 0.75
    "ARCHIVE_COLLECTED_DATA": 0.73,           # assumed 0.92
    "BRUTE_FORCE_SSH": 0.48,                  # assumed 0.60
    "CLEAR_LOGS": 0.69,                       # assumed 0.90
    "CREDENTIALS_FROM_BROWSER": 0.64,         # assumed 0.70
    "CREDS_IN_FILES": 0.44,                   # assumed 0.75
    "DATA_FROM_LOCAL_SYSTEM": 0.58,           # assumed 0.90
    "EXFILTRATE_DATA": 0.61,                  # assumed 1.00
    "EXTERNAL_REMOTE_SERVICES": 0.61,         # assumed 0.70
    "NETWORK_SCAN": 0.65,                     # assumed 0.95
    "PASSWORD_SPRAYING": 0.52,                # assumed 0.55
    "SQL_INJECTION": 0.51,                    # assumed 0.85
    "SYSTEM_INFO_DISCOVERY": 0.69,            # assumed 0.98
    "VALID_ACCOUNTS_LOGIN": 0.64,             # assumed 0.95
    "WEB_SHELL_UPLOAD": 0.59,                 # assumed 0.80
}

# mean of observed/assumed over all sixteen measurements
MEAN_RATIO = 0.732


def apply(techniques, mode="measured", scale=None):
    """Return {name: original_probability} so the caller can restore it.

    mode "measured"  substitute the measured rates for the techniques we could
                     execute, and leave the rest at their documented values.
    mode "scaled"    multiply every technique by `scale` (default MEAN_RATIO),
                     which asks what happens if the optimism we measured holds
                     across the whole catalogue.
    """
    original = {n: t.success_prob for n, t in techniques.items()}
    if mode == "measured":
        for n, p in MEASURED.items():
            if n in techniques:
                techniques[n].success_prob = p
    elif mode == "scaled":
        k = MEAN_RATIO if scale is None else scale
        for n, t in techniques.items():
            t.success_prob = max(0.05, min(1.0, t.success_prob * k))
    else:
        raise ValueError(f"unknown calibration mode {mode!r}")
    return original


def restore(techniques, original):
    for n, p in original.items():
        if n in techniques:
            techniques[n].success_prob = p
