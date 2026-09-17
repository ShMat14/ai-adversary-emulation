# env/state_models.py
from dataclasses import dataclass, field


@dataclass
class HostState:
    """Represents the security state of a single machine in the network.
    Using @dataclass so dataclasses.asdict() works in TelemetryLogger.
    """
    name: str
    compromised: bool = False
    has_credentials: bool = False
    privileged: bool = False
    detected: bool = False

    def reset(self):
        """Resets the host to a clean state."""
        self.compromised = False
        self.has_credentials = False
        self.privileged = False
        self.detected = False


class NetworkState:
    """Represents the entire network environment for the AI agent."""
    def __init__(self, host_names):
        # We use a dictionary for easy lookup by name: self.hosts["user01"]
        self.hosts = {name: HostState(name=name) for name in host_names}
        self.detection_triggered = False
        self.host_names = host_names  # Keep the order fixed for RL

    def reset(self):
        """Resets all hosts in the network."""
        for host in self.hosts.values():
            host.reset()
        self.detection_triggered = False

    def is_goal_achieved(self):
        """
        Thesis Logic: The goal is usually to compromise the
        Domain Controller (dc01) and gain admin privileges.
        """
        if "dc01" in self.hosts:
            return self.hosts["dc01"].privileged
        return False

    def any_host_detected(self):
        """Check if the Red Team has been spotted anywhere."""
        return any(host.detected for host in self.hosts.values())
