"""BC source gate; heuristic fog is approved (docs/modern/BC.md sections 10–11)."""

# These are dependencies of the current formatter, not missing values to fill with zero.
REPLAY_ONLY_BLOCKERS = (
    "cooldown normalizers and champion descriptors include sim-kit tables",
    "XP progress and skill points use simulator rules",
    "quest state and trinket charges inherit initial-state defaults",
    "missing scalar fields use unmarked defaults; the policy has no availability masks",
)


class ReplayOnlyUnavailable(ValueError):
    """The existing observation contract cannot represent replay-only inputs."""


def require_replay_only(operation: str):
    """Stop before reading/writing bulk data or training on old mixed-source arrays.

    Remove this hold only with a verified replay-only formatter and a loader that
    checks its per-feature provenance and consumes its availability masks.
    """
    raise ReplayOnlyUnavailable(
        f"{operation} is on hold: the current BC pipeline is not replay-only. "
        + "; ".join(REPLAY_ONLY_BLOCKERS)
        + ". See docs/modern/BC.md section 10. Existing v1-v5 arrays are not certified."
    )
