from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def wrap_angles(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def segment_point_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


@dataclass(frozen=True)
class ReacherConfig:
    link1: float = 0.65
    link2: float = 0.55
    dt: float = 0.08
    max_steps: int = 120
    success_radius: float = 0.09
    collision_penalty: float = 5.0
    success_bonus: float = 5.0
    progress_scale: float = 3.0
    action_penalty: float = 0.01
    clearance_penalty: float = 0.05
    clearance_margin: float = 0.08
    difficulty: str = "train"


class RandomizedReacherEnv(gym.Env):
    """A randomized 2-link safe-reaching task.

    The arm observes its joint angles, target, obstacle, current hand position,
    target vector, and current obstacle clearance. Each episode samples a new
    start pose, target, and obstacle. Collision terminates the episode and safe
    success requires reaching the target without collision.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: ReacherConfig | None = None):
        super().__init__()
        self.config = config or ReacherConfig()
        self.link_lengths = np.array([self.config.link1, self.config.link2], dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self.angles = np.zeros(2, dtype=np.float32)
        self.target = np.array([0.6, 0.5], dtype=np.float32)
        self.obstacle = np.array([0.4, 0.25], dtype=np.float32)
        self.obstacle_radius = 0.08
        self.steps = 0
        self.prev_distance = 0.0
        self.prev_clearance = 0.0

    def _points(self, angles: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self.angles if angles is None else angles
        base = np.array([0.0, 0.0], dtype=np.float32)
        elbow = self.link_lengths[0] * np.array(
            [math.cos(float(q[0])), math.sin(float(q[0]))],
            dtype=np.float32,
        )
        hand = elbow + self.link_lengths[1] * np.array(
            [math.cos(float(q[0] + q[1])), math.sin(float(q[0] + q[1]))],
            dtype=np.float32,
        )
        return base, elbow, hand

    def _distance_to_target(self) -> float:
        _, _, hand = self._points()
        return float(np.linalg.norm(hand - self.target))

    def _clearance(self, angles: np.ndarray | None = None, obstacle: np.ndarray | None = None) -> float:
        obs = self.obstacle if obstacle is None else obstacle
        base, elbow, hand = self._points(angles)
        d1 = segment_point_distance(base, elbow, obs)
        d2 = segment_point_distance(elbow, hand, obs)
        return min(d1, d2) - float(self.obstacle_radius)

    def _obs(self) -> np.ndarray:
        _, _, hand = self._points()
        target_vec = self.target - hand
        clearance = self._clearance()
        return np.array(
            [
                math.sin(float(self.angles[0])),
                math.cos(float(self.angles[0])),
                math.sin(float(self.angles[1])),
                math.cos(float(self.angles[1])),
                self.target[0],
                self.target[1],
                self.obstacle[0],
                self.obstacle[1],
                self.obstacle_radius,
                hand[0],
                hand[1],
                target_vec[0],
                target_vec[1],
                clearance,
            ],
            dtype=np.float32,
        )

    def _sample_target(self) -> np.ndarray:
        for _ in range(200):
            radius = self.rng.uniform(0.40, 0.95)
            theta = self.rng.uniform(0.25, 2.65)
            target = np.array([radius * math.cos(theta), radius * math.sin(theta)], dtype=np.float32)
            if target[1] > 0.05:
                return target
        return np.array([0.55, 0.55], dtype=np.float32)

    def _difficulty_params(self) -> tuple[tuple[float, float], tuple[float, float]]:
        difficulty = self.config.difficulty
        if difficulty == "easy":
            return (0.12, 0.28), (0.04, 0.07)
        if difficulty == "medium":
            return (0.02, 0.13), (0.06, 0.10)
        if difficulty == "hard":
            return (-0.03, 0.06), (0.075, 0.12)
        draw = self.rng.random()
        if draw < 0.45:
            return (-0.025, 0.06), (0.075, 0.115)
        if draw < 0.80:
            return (0.04, 0.14), (0.06, 0.10)
        return (0.14, 0.28), (0.045, 0.075)

    def _sample_scene(self) -> None:
        for _ in range(2000):
            direct_clearance_range, radius_range = self._difficulty_params()
            angles = np.array(
                [
                    self.rng.uniform(-0.9, 0.9),
                    self.rng.uniform(-1.1, 1.1),
                ],
                dtype=np.float32,
            )
            _, _, start_hand = self._points(angles)
            target = self._sample_target()
            start_to_target = float(np.linalg.norm(target - start_hand))
            if start_to_target < 0.25 or start_to_target > 0.95:
                continue

            direction = target - start_hand
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-6:
                continue
            unit = direction / norm
            perp = np.array([-unit[1], unit[0]], dtype=np.float32)
            obstacle_radius = float(self.rng.uniform(*radius_range))
            direct_path_clearance = float(self.rng.uniform(*direct_clearance_range))
            offset_mag = max(0.015, obstacle_radius + direct_path_clearance)
            offset_sign = -1.0 if self.rng.random() < 0.5 else 1.0
            along = self.rng.uniform(0.35, 0.70)
            obstacle = start_hand + along * direction + offset_sign * offset_mag * perp
            if obstacle[1] < -0.10 or np.linalg.norm(obstacle) > 1.15:
                continue

            self.angles = angles
            self.target = target
            self.obstacle = obstacle.astype(np.float32)
            self.obstacle_radius = obstacle_radius
            start_clearance = self._clearance()
            target_clearance = float(np.linalg.norm(target - obstacle) - obstacle_radius)
            if (
                start_clearance > 0.08
                and target_clearance > self.config.success_radius + 0.08
                and direct_path_clearance > direct_clearance_range[0] - 1e-9
            ):
                return

        self.angles = np.array([0.2, 0.2], dtype=np.float32)
        self.target = np.array([0.55, 0.55], dtype=np.float32)
        self.obstacle = np.array([0.55, 0.25], dtype=np.float32)
        self.obstacle_radius = 0.07

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._sample_scene()
        self.steps = 0
        self.prev_distance = self._distance_to_target()
        self.prev_clearance = self._clearance()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.steps += 1
        self.angles = wrap_angles(self.angles + action * self.config.dt).astype(np.float32)

        distance = self._distance_to_target()
        clearance = self._clearance()
        is_colliding = clearance <= 0.0
        safe_success = bool(distance <= self.config.success_radius and not is_colliding)
        raw_reach = bool(distance <= self.config.success_radius)
        unsafe_reach = bool(raw_reach and is_colliding)
        action_norm = float(np.linalg.norm(action))
        progress = self.prev_distance - distance
        clearance_shortfall = max(0.0, self.config.clearance_margin - clearance)
        progress_reward = self.config.progress_scale * progress
        success_bonus_reward = self.config.success_bonus if safe_success else 0.0
        collision_penalty_reward = -self.config.collision_penalty * float(is_colliding)
        action_penalty_reward = -self.config.action_penalty * (action_norm**2)
        clearance_penalty_reward = (
            -self.config.clearance_penalty * clearance_shortfall / max(self.config.clearance_margin, 1e-6)
        )

        reward = (
            progress_reward
            + success_bonus_reward
            + collision_penalty_reward
            + action_penalty_reward
            + clearance_penalty_reward
        )

        self.prev_distance = distance
        self.prev_clearance = clearance
        collision_terminated = bool(is_colliding and not safe_success)
        terminated = bool(safe_success or collision_terminated)
        truncated = self.steps >= self.config.max_steps

        info = {
            "distance_to_goal": distance,
            "is_colliding": float(is_colliding),
            "collision_terminated": float(collision_terminated),
            "success": float(safe_success),
            "safe_success": float(safe_success),
            "raw_reach": float(raw_reach),
            "unsafe_reach": float(unsafe_reach),
            "action_norm": action_norm,
            "obstacle_clearance": clearance,
            "progress_reward": float(progress_reward),
            "collision_penalty_reward": float(collision_penalty_reward),
            "success_bonus_reward": float(success_bonus_reward),
            "action_penalty_reward": float(action_penalty_reward),
            "clearance_penalty_reward": float(clearance_penalty_reward),
            "target_x": float(self.target[0]),
            "target_y": float(self.target[1]),
            "obstacle_x": float(self.obstacle[0]),
            "obstacle_y": float(self.obstacle[1]),
            "obstacle_radius": float(self.obstacle_radius),
        }
        return self._obs(), float(reward), terminated, truncated, info


def make_env(difficulty: str = "train", **kwargs: Any) -> RandomizedReacherEnv:
    config = ReacherConfig(difficulty=difficulty, **kwargs)
    return RandomizedReacherEnv(config)
