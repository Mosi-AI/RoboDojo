"""Reject discontinuous IK targets before any arm or gripper is dispatched.

The guard checks only proprioception and IK joint targets. It is not a
collision, contact, or task-progress detector. Periodic joint coordinates are
normalised to the nearest legal equivalent before continuity is measured.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np


def _nearest_legal_angles(target, reference, limits, periodic):
    """Map periodic targets to their closest in-bounds equivalent."""
    normalized = target.copy()
    valid = np.ones(target.shape, dtype=bool)
    for index in range(target.size):
        lower, upper = limits[index]
        if not periodic[index]:
            valid[index] = lower <= target[index] <= upper
            continue
        first_turn = int(np.ceil((lower - target[index]) / (2 * np.pi)))
        last_turn = int(np.floor((upper - target[index]) / (2 * np.pi)))
        if first_turn > last_turn:
            valid[index] = False
            continue
        nearest_turn = int(
            np.rint((reference[index] - target[index]) / (2 * np.pi))
        )
        turn = min(max(nearest_turn, first_turn), last_turn)
        normalized[index] = target[index] + turn * 2 * np.pi
    return normalized, valid


class IKContinuityError(ValueError):
    """Raised when an IK action fails the pre-dispatch continuity check."""

    def __init__(self, checks):
        self.checks = deepcopy(checks)
        self.event = {
            "type": "action_rejected_ik_continuity",
            "rejected_action_executed": False,
            "reason": (
                "IK targets failed joint continuity checks before dispatch. "
                "Neither arm nor gripper executed this action. Replan from "
                "the latest observed pose using smaller increments or another approach; "
                "do not compensate for the unexecuted suffix."
            ),
            "violations": [
                {
                    key: value
                    for key, value in check.items()
                    if key
                    not in (
                        "current_joints",
                        "target_joints",
                        "normalized_target_joints",
                        "previous_target",
                    )
                }
                for check in checks
                if not check["passed"]
            ],
        }
        super().__init__(self.event["reason"])


class IKContinuityGuard:
    """Check current-to-target and successive accepted-target increments.

    ``check`` has no side effects. ``commit`` is called only after every arm
    in a dual-arm action passes, so rejecting one arm also rejects its gripper.
    """

    def __init__(self, max_joint_increment_rad=0.15, log_path=None):
        if (
            isinstance(max_joint_increment_rad, bool)
            or not isinstance(max_joint_increment_rad, (int, float))
            or not np.isfinite(max_joint_increment_rad)
            or max_joint_increment_rad <= 0
        ):
            raise ValueError("IK joint increment must be finite and positive")
        self.limit = float(max_joint_increment_rad)
        self.previous_targets = {}
        self.log_path = Path(log_path) if log_path else None

    def check(self, candidates):
        if not isinstance(candidates, dict):
            raise ValueError("IK continuity requires a mapping of target arms")
        checks = []
        for arm, candidate in candidates.items():
            check = {"arm": arm, "passed": False, "limit_rad": self.limit}
            checks.append(check)
            if not isinstance(candidate, dict):
                check["error"] = "invalid_candidate"
                continue
            status = candidate.get("status")
            if status != "Success":
                # A solver-reported failure is distinct from malformed
                # candidate metadata, while both fail closed before dispatch.
                check["error"] = (
                    "ik_failed" if status is not None else "invalid_candidate"
                )
                continue
            try:
                current = np.asarray(candidate["current"], dtype=float)
                target = np.asarray(candidate["target"], dtype=float)
            except (ValueError, TypeError):
                check["error"] = "invalid_joint_vector"
                continue
            if (
                current.ndim != 1
                or not current.size
                or current.shape != target.shape
                or not np.isfinite(current).all()
                or not np.isfinite(target).all()
            ):
                check["error"] = "invalid_joint_vector"
                continue
            try:
                limits = np.asarray(candidate["joint_limits"], dtype=float)
                periodic = np.broadcast_to(
                    np.asarray(candidate["periodic"], dtype=bool), target.shape
                )
            except (KeyError, ValueError, TypeError):
                check["error"] = "invalid_joint_metadata"
                continue
            if (
                limits.shape != (target.size, 2)
                or not np.isfinite(limits).all()
                or np.any(limits[:, 0] > limits[:, 1])
            ):
                check["error"] = "invalid_joint_metadata"
                continue
            normalized, legal = _nearest_legal_angles(
                target, current, limits, periodic
            )
            check.update(
                current_joints=current.tolist(),
                target_joints=target.tolist(),
                normalized_target_joints=normalized.tolist(),
            )
            if not legal.all():
                check["error"] = "joint_target_out_of_bounds"
                check["out_of_bounds_joint_index"] = int(
                    np.flatnonzero(~legal)[0]
                )
                continue
            delta = np.abs(normalized - current)
            check.update(
                max_current_delta_rad=float(delta.max()),
                current_delta_joint_index=int(delta.argmax()),
            )
            previous = self.previous_targets.get(arm)
            if previous is not None:
                if previous.shape != target.shape:
                    check["error"] = "joint_dimension_changed"
                    continue
                previous_equivalent, previous_legal = _nearest_legal_angles(
                    target, previous, limits, periodic
                )
                if not previous_legal.all():
                    check["error"] = "joint_target_out_of_bounds"
                    check["out_of_bounds_joint_index"] = int(
                        np.flatnonzero(~previous_legal)[0]
                    )
                    continue
                delta_previous = np.abs(previous_equivalent - previous)
                check.update(
                    previous_target=previous.tolist(),
                    max_target_delta_rad=float(delta_previous.max()),
                    target_delta_joint_index=int(delta_previous.argmax()),
                )
            check["passed"] = (
                check["max_current_delta_rad"] <= self.limit + 1e-6
                and check.get("max_target_delta_rad", 0) <= self.limit + 1e-6
            )
            if not check["passed"]:
                check["error"] = "joint_target_jump"
        if not checks:
            raise ValueError("IK continuity requires target arm candidates")
        if any(not check["passed"] for check in checks):
            raise IKContinuityError(checks)
        return checks

    def commit(self, checks):
        self.previous_targets.update(
            {
                check["arm"]: np.asarray(
                    check["normalized_target_joints"], dtype=float
                ).copy()
                for check in checks
            }
        )

    def sync_joint_targets(self, targets):
        """Refresh the baseline after an accepted direct joint action or hold."""
        for arm, target in (targets or {}).items():
            try:
                value = np.asarray(target, dtype=float)
            except (TypeError, ValueError):
                continue
            if value.ndim == 1 and value.size and np.isfinite(value).all():
                self.previous_targets[arm] = value.copy()

    def record(self, step, checks, accepted):
        """Persist finite IK evidence, including rejected targets."""
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "attempted_step": step,
                        "accepted": accepted,
                        "checks": checks,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
