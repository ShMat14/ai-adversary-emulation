# env/ics_env.py
"""
Planned ICS/OT variant of the adversary environment (future work).

This module is a placeholder for the master's-level extension described in the
thesis: modelling an Industrial Control System network (PLCs, HMI, SCADA /
historian) whose objective is to manipulate a physical process value (e.g. a
water level) without tripping detection.

It is NOT yet implemented. To build it, subclass AdversaryEnv and override the
network construction and action-application logic with ICS-specific hosts and
techniques.
"""

from env.adversary_env import AdversaryEnv


class ICSAdversaryEnv(AdversaryEnv):
    """Future ICS-focused environment. Not yet implemented."""

    def __init__(self, config=None):
        raise NotImplementedError(
            "ICSAdversaryEnv is a placeholder for future work and is not yet implemented."
        )
