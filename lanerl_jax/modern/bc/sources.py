"""Versioned replay fields plus the approved fog heuristic; BC.md section 14."""
import json
from pathlib import Path

SOURCE_CONTRACT = "replay-fields-circles-missing-v1"


class ReplayOnlyUnavailable(ValueError):
    """Old mixed-source arrays cannot be used as replay-only inputs."""


def require_replay_only(operation: str, metadata=None):
    if isinstance(metadata, Path):
        path = metadata / "plan.json"
        metadata = json.loads(path.read_text()) if path.exists() else None
    profile = None
    if isinstance(metadata, dict):
        profile = metadata.get("profile", metadata.get("provenance", {}).get("obs_profile",
                                      metadata.get("profiles", {}).get("obs")))
    if (not isinstance(metadata, dict) or metadata.get("source_contract") != SOURCE_CONTRACT
            or profile != "modern-world-v4"):
        raise ReplayOnlyUnavailable(f"{operation}: missing replay source contract; legacy mixed-source arrays "
                                    "are not replay-only. See docs/modern/BC.md section 14.")
