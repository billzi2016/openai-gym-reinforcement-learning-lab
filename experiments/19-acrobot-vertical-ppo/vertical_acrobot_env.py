from __future__ import annotations

from math import cos, pi

import numpy as np
from gymnasium.envs.classic_control.acrobot import AcrobotEnv, wrap
from gymnasium.wrappers import TimeLimit


class VerticalAcrobotEnv(AcrobotEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        theta1_tolerance: float = 0.20,
        theta2_tolerance: float = 0.20,
        velocity_tolerance_1: float = 0.60,
        velocity_tolerance_2: float = 0.90,
        hold_steps: int = 10,
        damping_1: float = 0.01,
        damping_2: float = 0.10,
    ):
        super().__init__(render_mode=render_mode)
        self.theta1_tolerance = theta1_tolerance
        self.theta2_tolerance = theta2_tolerance
        self.velocity_tolerance_1 = velocity_tolerance_1
        self.velocity_tolerance_2 = velocity_tolerance_2
        self.hold_steps = hold_steps
        self.damping_1 = damping_1
        self.damping_2 = damping_2
        self.success_counter = 0
        self.previous_tip_height = -2.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.success_counter = 0
        obs, info = super().reset(seed=seed, options=options)
        self.previous_tip_height = self._pose_metrics()["tip_height"]
        return obs, info

    def _pose_metrics(self):
        s = self.state
        assert s is not None, "Call reset before using VerticalAcrobotEnv object."
        theta1, theta2, dtheta1, dtheta2 = s
        theta1_error = abs(wrap(theta1 - pi, -pi, pi))
        theta2_error = abs(wrap(theta2, -pi, pi))
        tip_height = -cos(theta1) - cos(theta1 + theta2)
        stable = (
            theta1_error <= self.theta1_tolerance
            and theta2_error <= self.theta2_tolerance
            and abs(dtheta1) <= self.velocity_tolerance_1
            and abs(dtheta2) <= self.velocity_tolerance_2
        )
        return {
            "theta1_error": theta1_error,
            "theta2_error": theta2_error,
            "dtheta1": dtheta1,
            "dtheta2": dtheta2,
            "tip_height": tip_height,
            "stable": stable,
        }

    def step(self, a):
        obs, _, _, _, _ = super().step(a)
        metrics = self._pose_metrics()
        tip_height = metrics["tip_height"]
        height_gain = tip_height - self.previous_tip_height
        self.previous_tip_height = tip_height

        upright_gate = float(np.clip((tip_height - 0.7) / 1.1, 0.0, 1.0))
        top_gate = float(np.clip((tip_height - 1.45) / 0.35, 0.0, 1.0))
        alignment_penalty = metrics["theta1_error"] ** 2 + 0.60 * metrics["theta2_error"] ** 2
        low_speed_penalty = 0.0025 * (metrics["dtheta1"] ** 2) + 0.0010 * (metrics["dtheta2"] ** 2)
        top_speed_penalty = 0.0500 * (metrics["dtheta1"] ** 2) + 0.0200 * (metrics["dtheta2"] ** 2)
        velocity_penalty = (1.0 - upright_gate) * low_speed_penalty + upright_gate * top_speed_penalty
        action_penalty = 0.008 * abs(self.AVAIL_TORQUE[a])

        reward = (
            7.0 * height_gain
            + 0.70 * tip_height
            + upright_gate * (2.8 - 1.35 * alignment_penalty)
            + top_gate * (4.0 - 1.10 * alignment_penalty)
            - velocity_penalty
            - action_penalty
            - 0.03
        )

        if tip_height >= 1.6:
            reward += 2.0
        if tip_height >= 1.85:
            reward += 4.0

        if metrics["stable"]:
            self.success_counter += 1
            reward += 14.0 + 2.0 * self.success_counter
        else:
            self.success_counter = 0

        terminated = self.success_counter >= self.hold_steps
        if terminated:
            reward += 260.0

        info = {
            "tip_height": float(metrics["tip_height"]),
            "height_gain": float(height_gain),
            "upright_gate": float(upright_gate),
            "theta1_error": float(metrics["theta1_error"]),
            "theta2_error": float(metrics["theta2_error"]),
            "dtheta1": float(metrics["dtheta1"]),
            "dtheta2": float(metrics["dtheta2"]),
            "success_counter": int(self.success_counter),
            "is_vertical_success": bool(terminated),
        }
        return obs, float(reward), terminated, False, info

    def _dsdt(self, s_augmented):
        derivatives = list(super()._dsdt(s_augmented))
        s = self.state
        assert s is not None, "Call reset before using VerticalAcrobotEnv object."
        dtheta1 = s[2]
        dtheta2 = s[3]
        derivatives[2] -= self.damping_1 * dtheta1
        derivatives[3] -= self.damping_2 * dtheta2
        return tuple(derivatives)


def make_vertical_acrobot_env(
    *,
    render_mode: str | None = None,
    max_episode_steps: int = 600,
    theta1_tolerance: float = 0.20,
    theta2_tolerance: float = 0.20,
    velocity_tolerance_1: float = 0.60,
    velocity_tolerance_2: float = 0.90,
    hold_steps: int = 10,
    damping_1: float = 0.01,
    damping_2: float = 0.10,
):
    env = VerticalAcrobotEnv(
        render_mode=render_mode,
        theta1_tolerance=theta1_tolerance,
        theta2_tolerance=theta2_tolerance,
        velocity_tolerance_1=velocity_tolerance_1,
        velocity_tolerance_2=velocity_tolerance_2,
        hold_steps=hold_steps,
        damping_1=damping_1,
        damping_2=damping_2,
    )
    return TimeLimit(env, max_episode_steps=max_episode_steps)
