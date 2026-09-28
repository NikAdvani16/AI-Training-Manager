from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from env import ReacherConfig, make_env
from manager import BOUNDS, TrainingRecipe, allowed_actions, apply_recipe_update


ROOT = Path(__file__).resolve().parent
RECIPE_KEYS = {
    "actor_lr",
    "critic_lr",
    "entropy_coef",
    "clip_coef",
    "update_epochs",
    "target_kl",
    "initial_actor_logstd",
    "collision_penalty",
    "success_bonus",
    "progress_scale",
}


def verify_configs() -> None:
    common = json.loads((ROOT / "configs" / "common.json").read_text(encoding="utf-8"))
    assert common["reported_result_seeds"] == [40, 41, 42]
    assert common["environment"]["collision_terminates_episode"] is True
    assert common["environment"]["safe_success_requires_no_collision"] is True
    assert common["ppo"]["anneal_lr"] is False
    env_defaults = ReacherConfig()
    for key in ("link1", "link2", "dt", "action_penalty", "clearance_penalty", "clearance_margin"):
        assert common["environment"][key] == getattr(env_defaults, key), (
            f"common environment config and ReacherConfig disagree on {key}"
        )
    for name in ("baseline", "conservative", "aggressive"):
        payload = json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
        recipe = payload["recipe"]
        assert set(recipe) == RECIPE_KEYS, f"{name}: recipe keys differ from the release schema"
        managed = dict(recipe)
        managed.pop("initial_actor_logstd")
        parsed = TrainingRecipe(**managed)
        before = asdict(parsed)
        parsed.clamp()
        assert asdict(parsed) == before, f"{name}: a configured value is outside manager bounds"


def verify_environment() -> None:
    env_a = make_env(difficulty="train")
    env_b = make_env(difficulty="train")
    obs_a, _ = env_a.reset(seed=1234)
    obs_b, _ = env_b.reset(seed=1234)
    assert np.array_equal(obs_a, obs_b), "same environment seed did not reproduce the same scene"
    obs_c, _ = env_b.reset(seed=1235)
    assert not np.array_equal(obs_a, obs_c), "different environment seeds produced the same scene"
    _, _, terminated, truncated, info = env_a.step(np.zeros(2, dtype=np.float32))
    assert not (terminated and truncated)
    assert {"safe_success", "collision_terminated", "distance_to_goal"} <= set(info)


def verify_action_surface() -> None:
    recipe = TrainingRecipe(3e-4, 1e-3, 0.01, 0.2, 6, 0.03, 5.0, 5.0, 3.0)
    surface = allowed_actions(recipe, allow_rollback=True, allow_stop=True)
    assert "request_extra_eval" not in surface
    for action, spec in surface.items():
        if action in {"no_action", "rollback_to_best", "stop_training"}:
            continue
        choices = spec.get("exactly_one_of", spec.get("two_or_three_of", spec))
        for key, values in choices.items():
            for value in values:
                changed = apply_recipe_update(recipe, {key: value})
                assert asdict(changed) != asdict(recipe), f"no-op offered: {action}.{key}={value}"
    for key, (low, high) in BOUNDS.items():
        assert low <= high, f"invalid bound for {key}"


def main() -> None:
    verify_configs()
    verify_environment()
    verify_action_surface()
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=ROOT,
        check=True,
    )
    print("Release verification passed: configs, environment determinism, action surface, prompts, verifier, and rollback tests.")


if __name__ == "__main__":
    main()
