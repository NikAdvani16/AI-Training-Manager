from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

try:
    from .env import ReacherConfig, make_env
    from .manager import (
        DEFAULT_FORMATTER_MODEL,
        DEFAULT_FORMATTER_TEMPERATURE,
        DEFAULT_LLM_MODEL,
        DEFAULT_TEMPERATURE,
        RandomizedReacherManager,
        TrainingRecipe,
        allowed_actions,
        apply_recipe_update,
    )
except ImportError:  # Direct script execution from the release repository.
    from env import ReacherConfig, make_env
    from manager import (
        DEFAULT_FORMATTER_MODEL,
        DEFAULT_FORMATTER_TEMPERATURE,
        DEFAULT_LLM_MODEL,
        DEFAULT_TEMPERATURE,
        RandomizedReacherManager,
        TrainingRecipe,
        allowed_actions,
        apply_recipe_update,
    )


def pick_device(preferred: str = "auto") -> torch.device:
    if preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested with --device mps, but torch.backends.mps.is_available() is false."
            )
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, initial_logstd: float):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), float(initial_logstd)))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(obs)
        return action, logprob, entropy, value


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true, unbiased=False)
    if float(var_y.detach().cpu()) <= 1e-12:
        return 0.0
    return float((1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y).detach().cpu())


def actor_parameter_vector(model: ActorCritic) -> torch.Tensor:
    actor_params = [*model.actor_mean.parameters(), model.actor_logstd]
    return nn.utils.parameters_to_vector(actor_params).detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO baseline for randomized safe-reaching.")
    parser.add_argument("--out_dir", type=str, default="runs/baseline_seed1")
    parser.add_argument("--mode", choices=["fixed", "ai_manager"], default="fixed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_updates", type=int, default=500)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps_per_update", type=int, default=128)
    parser.add_argument("--max_episode_steps", type=int, default=120)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--initial_actor_logstd", type=float, default=-0.7)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--update_epochs", type=int, default=6)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=80)
    parser.add_argument(
        "--final_eval_episodes",
        type=int,
        default=240,
        help="Held-out final-test episodes per difficulty. These are not shown to the manager.",
    )
    parser.add_argument(
        "--final_eval_seed_base",
        type=int,
        default=9_000_000,
        help="Seed base for the held-out final-test scene set.",
    )
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="Skip the final held-out evaluation, useful only for quick smoke tests.",
    )
    parser.add_argument("--success_radius", type=float, default=0.09)
    parser.add_argument("--collision_penalty", type=float, default=5.0)
    parser.add_argument("--success_bonus", type=float, default=5.0)
    parser.add_argument("--progress_scale", type=float, default=3.0)
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm_temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--llm_formatter_model", type=str, default=DEFAULT_FORMATTER_MODEL)
    parser.add_argument("--llm_formatter_temperature", type=float, default=DEFAULT_FORMATTER_TEMPERATURE)
    parser.add_argument("--async_manager_lead_updates", type=int, default=5)
    # After this many updates without a decision, training pauses and blocks on the
    # manager call instead of discarding it, so a response is never wasted.
    parser.add_argument("--async_manager_max_wait_updates", type=int, default=5)
    # Safety valve: if the blocking wait exceeds this, the call is abandoned as stale
    # so a hung API cannot stall training indefinitely.
    parser.add_argument("--async_manager_wait_timeout_seconds", type=float, default=600.0)
    parser.add_argument("--rollback_min_drop", type=float, default=0.10)
    parser.add_argument("--rollback_patience", type=int, default=2)
    parser.add_argument("--rollback_cooldown_intervals", type=int, default=2)
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def evaluate(
    model: ActorCritic,
    device: torch.device,
    difficulty: str,
    episodes: int,
    seed: int,
    max_episode_steps: int,
    success_radius: float,
    collision_penalty: float,
    success_bonus: float,
    progress_scale: float,
    deterministic: bool = True,
) -> dict[str, float]:
    env = make_env(
        difficulty=difficulty,
        max_steps=max_episode_steps,
        success_radius=success_radius,
        collision_penalty=collision_penalty,
        success_bonus=success_bonus,
        progress_scale=progress_scale,
    )
    was_training = model.training
    model.eval()
    cpu_rng_state = torch.random.get_rng_state()
    accelerator_rng_state = None
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        try:
            accelerator_rng_state = torch.mps.get_rng_state()
        except Exception:  # noqa: BLE001 - some torch builds do not expose MPS RNG state cleanly.
            accelerator_rng_state = None
    returns: list[float] = []
    safe_successes: list[float] = []
    collisions: list[float] = []
    min_distances: list[float] = []
    final_distances: list[float] = []
    action_norms: list[float] = []
    action_saturation_fractions: list[float] = []
    lengths: list[float] = []
    steps_to_success: list[float] = []
    progress_rewards: list[float] = []
    collision_penalty_rewards: list[float] = []
    success_bonus_rewards: list[float] = []
    try:
        with torch.no_grad():
            for ep in range(episodes):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                ep_return = 0.0
                ep_collision = 0.0
                ep_safe = 0.0
                ep_distances: list[float] = []
                ep_actions: list[float] = []
                ep_action_saturation: list[float] = []
                ep_progress_reward = 0.0
                ep_collision_penalty_reward = 0.0
                ep_success_bonus_reward = 0.0
                steps = 0
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    if deterministic:
                        action_t = model.actor_mean(obs_t)
                    else:
                        action_t, _, _, _ = model.get_action_and_value(obs_t)
                    sampled_action = action_t.squeeze(0).cpu().numpy()
                    ep_action_saturation.append(float(np.mean(np.abs(sampled_action) > 1.0)))
                    obs, reward, terminated, truncated, info = env.step(sampled_action)
                    done = bool(terminated or truncated)
                    steps += 1
                    ep_return += float(reward)
                    ep_collision = max(ep_collision, float(info["is_colliding"]))
                    ep_safe = max(ep_safe, float(info["safe_success"]))
                    ep_distances.append(float(info["distance_to_goal"]))
                    ep_actions.append(float(info["action_norm"]))
                    ep_progress_reward += float(info.get("progress_reward", 0.0))
                    ep_collision_penalty_reward += float(info.get("collision_penalty_reward", 0.0))
                    ep_success_bonus_reward += float(info.get("success_bonus_reward", 0.0))
                returns.append(ep_return)
                collisions.append(ep_collision)
                safe_successes.append(ep_safe)
                if ep_safe > 0.0:
                    steps_to_success.append(float(steps))
                min_distances.append(float(np.min(ep_distances)) if ep_distances else 0.0)
                final_distances.append(float(ep_distances[-1]) if ep_distances else 0.0)
                action_norms.append(float(np.mean(ep_actions)) if ep_actions else 0.0)
                action_saturation_fractions.append(
                    float(np.mean(ep_action_saturation)) if ep_action_saturation else 0.0
                )
                lengths.append(float(steps))
                progress_rewards.append(ep_progress_reward)
                collision_penalty_rewards.append(ep_collision_penalty_reward)
                success_bonus_rewards.append(ep_success_bonus_reward)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if accelerator_rng_state is not None and hasattr(torch.mps, "set_rng_state"):
            try:
                torch.mps.set_rng_state(accelerator_rng_state)
            except Exception:  # noqa: BLE001
                pass
        if was_training:
            model.train()
    return {
        "safe_success_rate": float(np.mean(safe_successes)),
        "collision_episode_rate": float(np.mean(collisions)),
        "mean_return": float(np.mean(returns)),
        "mean_min_distance": float(np.mean(min_distances)),
        "mean_final_distance": float(np.mean(final_distances)),
        "mean_action_norm": float(np.mean(action_norms)),
        "sampled_action_saturation_fraction": float(np.mean(action_saturation_fractions)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_steps_to_success": float(np.mean(steps_to_success)) if steps_to_success else float("nan"),
        "mean_progress_reward": float(np.mean(progress_rewards)),
        "mean_collision_penalty_reward": float(np.mean(collision_penalty_rewards)),
        "mean_success_bonus_reward": float(np.mean(success_bonus_rewards)),
    }


def write_metrics_header(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_metrics_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def apply_reward_recipe_to_envs(envs: list[Any], recipe: TrainingRecipe) -> None:
    for env in envs:
        env.config = replace(
            env.config,
            collision_penalty=recipe.collision_penalty,
            success_bonus=recipe.success_bonus,
            progress_scale=recipe.progress_scale,
        )


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    recipe: TrainingRecipe,
    config: dict[str, Any],
    update: int,
    global_step: int,
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "recipe": asdict(recipe),
            "config": config,
            "update": update,
            "global_step": global_step,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    recipe_override: TrainingRecipe | None = None,
) -> tuple[TrainingRecipe, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_to_device(optimizer, device)
    recipe_source = asdict(recipe_override) if recipe_override is not None else checkpoint["recipe"]
    recipe = TrainingRecipe(**recipe_source)
    recipe.clamp()
    set_optimizer_learning_rates(optimizer, recipe)
    return recipe, checkpoint


def set_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    recipe: TrainingRecipe,
) -> None:
    optimizer.param_groups[0]["lr"] = recipe.actor_lr
    optimizer.param_groups[1]["lr"] = recipe.critic_lr


def prefix(prefix_text: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix_text}_{key}": value for key, value in values.items()}


def split_eval_summary(evals: dict[str, dict[str, float]]) -> dict[str, float]:
    difficulties = ("easy", "medium", "hard")
    step_values = [
        evals[d]["mean_steps_to_success"]
        for d in difficulties
        if not math.isnan(float(evals[d]["mean_steps_to_success"]))
    ]
    return {
        "safe_success_mean": float(np.mean([evals[d]["safe_success_rate"] for d in difficulties])),
        "collision_mean": float(np.mean([evals[d]["collision_episode_rate"] for d in difficulties])),
        "min_distance_mean": float(np.mean([evals[d]["mean_min_distance"] for d in difficulties])),
        "final_distance_mean": float(np.mean([evals[d]["mean_final_distance"] for d in difficulties])),
        "episode_length_mean": float(np.mean([evals[d]["mean_episode_length"] for d in difficulties])),
        "steps_to_success_mean": float(np.mean(step_values)) if step_values else float("nan"),
        "progress_reward_mean": float(np.mean([evals[d]["mean_progress_reward"] for d in difficulties])),
        "collision_penalty_reward_mean": float(
            np.mean([evals[d]["mean_collision_penalty_reward"] for d in difficulties])
        ),
        "success_bonus_reward_mean": float(np.mean([evals[d]["mean_success_bonus_reward"] for d in difficulties])),
    }


CHECKPOINT_SELECTION_CRITERION = (
    "worst safe-success rate minus worst collision rate across deterministic/stochastic "
    "easy/medium/hard evaluation cells; ties prefer higher worst-cell safe success, "
    "lower worst-cell collision, higher mean utility, then lower deterministic final distance"
)


def checkpoint_robust_metrics(row: dict[str, Any]) -> dict[str, float | str]:
    det_safe = float(row["eval_safe_success_mean"])
    stochastic_safe = float(row["stochastic_eval_safe_success_mean"])
    det_collision = float(row["eval_collision_mean"])
    stochastic_collision = float(row["stochastic_eval_collision_mean"])
    safe_cells = [
        float(row.get(f"eval_{difficulty}_safe_success", det_safe))
        for difficulty in ("easy", "medium", "hard")
    ] + [
        float(row.get(f"stochastic_eval_{difficulty}_safe_success", stochastic_safe))
        for difficulty in ("easy", "medium", "hard")
    ]
    collision_cells = [
        float(row.get(f"eval_{difficulty}_collision", det_collision))
        for difficulty in ("easy", "medium", "hard")
    ] + [
        float(row.get(f"stochastic_eval_{difficulty}_collision", stochastic_collision))
        for difficulty in ("easy", "medium", "hard")
    ]
    mean_safe = 0.5 * (det_safe + stochastic_safe)
    mean_collision = 0.5 * (det_collision + stochastic_collision)
    worst_safe = min(safe_cells)
    worst_collision = max(collision_cells)
    return {
        "checkpoint_selection_criterion": CHECKPOINT_SELECTION_CRITERION,
        "checkpoint_robust_score": worst_safe - worst_collision,
        "checkpoint_mean_utility": mean_safe - mean_collision,
        "checkpoint_mean_safe_success": mean_safe,
        "checkpoint_worst_eval_cell_safe_success": worst_safe,
        "checkpoint_mean_collision": mean_collision,
        "checkpoint_worst_eval_cell_collision": worst_collision,
    }


def checkpoint_selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["checkpoint_robust_score"]),
        float(row["checkpoint_worst_eval_cell_safe_success"]),
        -float(row["checkpoint_worst_eval_cell_collision"]),
        float(row["checkpoint_mean_utility"]),
        -float(row["eval_final_distance_mean"]),
    )


def meets_robust_stop_criterion(row: dict[str, Any]) -> bool:
    required_keys = [
        f"{prefix}_{difficulty}_{metric}"
        for prefix in ("eval", "stochastic_eval")
        for difficulty in ("easy", "medium", "hard")
        for metric in ("safe_success", "collision")
    ]
    if not all(isinstance(row.get(key), (int, float)) for key in required_keys):
        return False
    return all(
        float(row[f"eval_{difficulty}_safe_success"]) >= 0.85
        and float(row[f"stochastic_eval_{difficulty}_safe_success"]) >= 0.85
        and float(row[f"eval_{difficulty}_collision"]) <= 0.15
        and float(row[f"stochastic_eval_{difficulty}_collision"]) <= 0.15
        for difficulty in ("easy", "medium", "hard")
    )


def update_row_from_evals(
    row: dict[str, Any],
    evals: dict[str, dict[str, float]],
    stochastic_evals: dict[str, dict[str, float]],
) -> None:
    safe_values = [evals[d]["safe_success_rate"] for d in ("easy", "medium", "hard")]
    collision_values = [evals[d]["collision_episode_rate"] for d in ("easy", "medium", "hard")]
    min_distance_values = [evals[d]["mean_min_distance"] for d in ("easy", "medium", "hard")]
    det_summary = split_eval_summary(evals)
    stochastic_summary = split_eval_summary(stochastic_evals)
    row.update(
        {
            "eval_safe_success_mean": float(np.mean(safe_values)),
            "eval_collision_mean": float(np.mean(collision_values)),
            "eval_min_distance_mean": float(np.mean(min_distance_values)),
            "eval_final_distance_mean": det_summary["final_distance_mean"],
            "eval_episode_length_mean": det_summary["episode_length_mean"],
            "eval_steps_to_success_mean": det_summary["steps_to_success_mean"],
            "eval_progress_reward_mean": det_summary["progress_reward_mean"],
            "eval_collision_penalty_reward_mean": det_summary["collision_penalty_reward_mean"],
            "eval_success_bonus_reward_mean": det_summary["success_bonus_reward_mean"],
            "stochastic_eval_safe_success_mean": stochastic_summary["safe_success_mean"],
            "stochastic_eval_collision_mean": stochastic_summary["collision_mean"],
            "stochastic_eval_easy_safe_success": stochastic_evals["easy"]["safe_success_rate"],
            "stochastic_eval_medium_safe_success": stochastic_evals["medium"]["safe_success_rate"],
            "stochastic_eval_hard_safe_success": stochastic_evals["hard"]["safe_success_rate"],
            "stochastic_eval_easy_collision": stochastic_evals["easy"]["collision_episode_rate"],
            "stochastic_eval_medium_collision": stochastic_evals["medium"]["collision_episode_rate"],
            "stochastic_eval_hard_collision": stochastic_evals["hard"]["collision_episode_rate"],
            "eval_easy_safe_success": evals["easy"]["safe_success_rate"],
            "eval_medium_safe_success": evals["medium"]["safe_success_rate"],
            "eval_hard_safe_success": evals["hard"]["safe_success_rate"],
            "eval_easy_collision": evals["easy"]["collision_episode_rate"],
            "eval_medium_collision": evals["medium"]["collision_episode_rate"],
            "eval_hard_collision": evals["hard"]["collision_episode_rate"],
            "eval_easy_min_distance": evals["easy"]["mean_min_distance"],
            "eval_medium_min_distance": evals["medium"]["mean_min_distance"],
            "eval_hard_min_distance": evals["hard"]["mean_min_distance"],
            "eval_easy_final_distance": evals["easy"]["mean_final_distance"],
            "eval_medium_final_distance": evals["medium"]["mean_final_distance"],
            "eval_hard_final_distance": evals["hard"]["mean_final_distance"],
            "eval_easy_episode_length": evals["easy"]["mean_episode_length"],
            "eval_medium_episode_length": evals["medium"]["mean_episode_length"],
            "eval_hard_episode_length": evals["hard"]["mean_episode_length"],
            "eval_easy_steps_to_success": evals["easy"]["mean_steps_to_success"],
            "eval_medium_steps_to_success": evals["medium"]["mean_steps_to_success"],
            "eval_hard_steps_to_success": evals["hard"]["mean_steps_to_success"],
            "eval_easy_progress_reward": evals["easy"]["mean_progress_reward"],
            "eval_medium_progress_reward": evals["medium"]["mean_progress_reward"],
            "eval_hard_progress_reward": evals["hard"]["mean_progress_reward"],
            "eval_easy_collision_penalty_reward": evals["easy"]["mean_collision_penalty_reward"],
            "eval_medium_collision_penalty_reward": evals["medium"]["mean_collision_penalty_reward"],
            "eval_hard_collision_penalty_reward": evals["hard"]["mean_collision_penalty_reward"],
            "eval_easy_success_bonus_reward": evals["easy"]["mean_success_bonus_reward"],
            "eval_medium_success_bonus_reward": evals["medium"]["mean_success_bonus_reward"],
            "eval_hard_success_bonus_reward": evals["hard"]["mean_success_bonus_reward"],
        }
    )
    row.update(checkpoint_robust_metrics(row))


def run_eval_suite(
    model: ActorCritic,
    device: torch.device,
    *,
    eval_episodes: int,
    seed_base: int,
    max_episode_steps: int,
    success_radius: float,
    collision_penalty: float,
    success_bonus: float,
    progress_scale: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    evals: dict[str, dict[str, float]] = {}
    stochastic_evals: dict[str, dict[str, float]] = {}
    offsets = {"easy": 0, "medium": 100, "hard": 200}
    for difficulty in ("easy", "medium", "hard"):
        offset = offsets[difficulty]
        evals[difficulty] = evaluate(
            model,
            device,
            difficulty=difficulty,
            episodes=eval_episodes,
            seed=seed_base + offset,
            max_episode_steps=max_episode_steps,
            success_radius=success_radius,
            collision_penalty=collision_penalty,
            success_bonus=success_bonus,
            progress_scale=progress_scale,
            deterministic=True,
        )
        stochastic_evals[difficulty] = evaluate(
            model,
            device,
            difficulty=difficulty,
            episodes=eval_episodes,
            seed=seed_base + offset,
            max_episode_steps=max_episode_steps,
            success_radius=success_radius,
            collision_penalty=collision_penalty,
            success_bonus=success_bonus,
            progress_scale=progress_scale,
            deterministic=False,
        )
    return evals, stochastic_evals


def write_final_heldout_eval(
    out_dir: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    recipe: TrainingRecipe,
    config: dict[str, Any],
    completed_update: int,
    global_step: int,
    best_checkpoint_path: Path,
    best_checkpoint_metrics: dict[str, Any] | None,
    final_eval_episodes: int,
    final_eval_seed_base: int,
    max_episode_steps: int,
    success_radius: float,
) -> None:
    if final_eval_episodes <= 0:
        return

    def evaluate_policy(policy_name: str, policy_recipe: TrainingRecipe) -> dict[str, Any]:
        deterministic_evals, stochastic_evals = run_eval_suite(
            model,
            device,
            eval_episodes=final_eval_episodes,
            seed_base=final_eval_seed_base,
            max_episode_steps=max_episode_steps,
            success_radius=success_radius,
            collision_penalty=policy_recipe.collision_penalty,
            success_bonus=policy_recipe.success_bonus,
            progress_scale=policy_recipe.progress_scale,
        )
        deterministic_summary = split_eval_summary(deterministic_evals)
        stochastic_summary = split_eval_summary(stochastic_evals)
        return {
            "policy_name": policy_name,
            "deterministic_summary": deterministic_summary,
            "stochastic_summary": stochastic_summary,
            "deterministic_by_difficulty": deterministic_evals,
            "stochastic_by_difficulty": stochastic_evals,
            "recipe": asdict(policy_recipe),
        }

    payload: dict[str, Any] = {
        "note": (
            "Held-out final evaluation. These episodes are not used for manager decisions, "
            "checkpoint selection, rollback, or training-time diagnostics."
        ),
        "final_eval_seed_base": final_eval_seed_base,
        "final_eval_episodes_per_difficulty": final_eval_episodes,
        "stochastic_final_eval_episodes_per_difficulty": final_eval_episodes,
        "completed_update": completed_update,
        "global_step": global_step,
        "config": config,
        "policies": {},
    }

    payload["policies"]["final_policy"] = evaluate_policy("final_policy", recipe)

    if best_checkpoint_path.exists():
        active_state = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
        }
        best_recipe, checkpoint = load_checkpoint(best_checkpoint_path, model, optimizer, device)
        best_payload = evaluate_policy("best_checkpoint_policy", best_recipe)
        best_payload["checkpoint_update"] = checkpoint.get("update")
        best_payload["checkpoint_global_step"] = checkpoint.get("global_step")
        best_payload["diagnostic_selection_metrics"] = best_checkpoint_metrics or checkpoint.get("metrics", {})
        payload["policies"]["best_checkpoint_policy"] = best_payload
        model.load_state_dict(active_state["model"])
        optimizer.load_state_dict(active_state["optimizer"])
        optimizer_to_device(optimizer, device)

    (out_dir / "final_heldout_eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    fieldnames = [
        "policy_name",
        "difficulty",
        "eval_type",
        "safe_success_rate",
        "collision_episode_rate",
        "mean_return",
        "mean_min_distance",
        "mean_final_distance",
        "mean_action_norm",
        "sampled_action_saturation_fraction",
        "mean_episode_length",
        "mean_steps_to_success",
        "mean_progress_reward",
        "mean_collision_penalty_reward",
        "mean_success_bonus_reward",
    ]
    with (out_dir / "final_heldout_eval_by_difficulty.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for policy in payload["policies"].values():
            for eval_type, eval_key in (
                ("deterministic", "deterministic_by_difficulty"),
                ("stochastic", "stochastic_by_difficulty"),
            ):
                for difficulty, metrics in policy[eval_key].items():
                    writer.writerow(
                        {
                            "policy_name": policy["policy_name"],
                            "difficulty": difficulty,
                            "eval_type": eval_type,
                            **{key: metrics.get(key, "") for key in fieldnames if key not in {"policy_name", "difficulty", "eval_type"}},
                        }
                    )


def build_manager_context(
    *,
    update: int,
    global_step: int,
    recipe: TrainingRecipe,
    latest: dict[str, Any],
    deterministic_evals: dict[str, dict[str, float]],
    stochastic_evals: dict[str, dict[str, float]],
    telemetry_history: list[dict[str, Any]],
    decision_history: list[dict[str, Any]],
    best_checkpoint_metrics: dict[str, Any] | None,
    rollback_gate: dict[str, Any],
    run_plan: dict[str, Any],
) -> dict[str, Any]:
    det_summary = split_eval_summary(deterministic_evals)
    stoch_summary = split_eval_summary(stochastic_evals)
    best_robust_score = float((best_checkpoint_metrics or {}).get("checkpoint_robust_score", 0.0))
    current_robust_score = float(latest.get("checkpoint_robust_score", 0.0))
    current_meets_stop_criterion = meets_robust_stop_criterion(latest)
    prior_eval_rows = [
        row
        for row in telemetry_history
        if isinstance(row.get("eval_easy_safe_success"), (int, float))
    ]
    prior_meets_stop_criterion = bool(prior_eval_rows) and meets_robust_stop_criterion(prior_eval_rows[-1])
    stop_allowed = current_meets_stop_criterion and prior_meets_stop_criterion
    action_surface = allowed_actions(
        recipe,
        allow_rollback=bool(best_checkpoint_metrics) and bool(rollback_gate.get("gate_passed", False)),
        allow_stop=stop_allowed,
    )
    return {
        "manager_version": "rl_training_manager_release_v1",
        "run_plan": run_plan,
        "task": {
            "task_family": "episodic continuous-control reinforcement learning",
            "environment_summary": "2-link arm reaches randomized targets while avoiding randomized circular obstacles",
            "success_definition": "safe reach only: target contact without collision",
            "episode_randomization": "start joint angles, target location, obstacle location, and obstacle radius are randomized every episode",
            "primary_objective": "maximize safe success on fixed randomized easy/medium/hard diagnostic splits while minimizing collision and PPO drift",
        },
        "algorithm": {
            "name": "PPO",
            "policy_type": "Gaussian actor with learned log standard deviation",
            "important_constraint": "actor_logstd is learned by PPO and is telemetry only; the manager may not edit it directly",
        },
        "fixed_task_parameters": {
            "not_manager_controls": [
                "collision termination",
                "success radius",
                "target sampling distribution",
                "obstacle sampling distribution",
                "model architecture",
                "optimizer type",
                "learned actor_logstd",
            ],
            "manager_controlled_reward_weights": [
                "collision_penalty",
                "success_bonus",
                "progress_scale",
            ],
        },
        "metric_definitions": {
            "train_safe_success_rate": "recent PPO rollout safe success",
            "eval_safe_success_mean": "mean deterministic safe success across easy/medium/hard eval",
            "stochastic_eval_safe_success_mean": "mean sampled-policy safe success across easy/medium/hard eval",
            "eval_collision_mean": "mean deterministic collision episode rate across eval splits",
            "stochastic_eval_collision_mean": "mean sampled-policy collision episode rate across eval splits",
            "eval_final_distance_mean": "mean deterministic final distance to target across eval splits",
            "eval_episode_length_mean": "mean deterministic episode length across eval splits",
            "eval_steps_to_success_mean": "mean deterministic steps to safe success, computed only over successful episodes",
            "eval_progress_reward_mean": "mean deterministic cumulative progress-reward contribution per episode",
            "eval_collision_penalty_reward_mean": "mean deterministic cumulative collision-penalty contribution per episode",
            "eval_success_bonus_reward_mean": "mean deterministic cumulative success-bonus contribution per episode",
            "actor_logstd": "learned policy log standard deviation; telemetry only, not a direct manager control",
            "approx_kl": "PPO update divergence indicator",
            "clip_fraction": "fraction of PPO samples affected by clipping",
            "collision_penalty": "training reward penalty applied when an episode collides; evaluation success still requires no collision",
            "success_bonus": "training reward bonus for safe success",
            "progress_scale": "training reward scale for distance-to-target progress",
            "configured_update_epochs": "PPO epochs requested by the active recipe",
            "epochs_ran": "PPO epochs actually completed before target_kl early stopping",
            "epoch_utilization": "epochs_ran divided by configured_update_epochs; low values mean KL early stopping limited the update",
            "kl_to_target": "approx_kl divided by target_kl; values near or above 1 mean the KL limit is binding",
            "actor_action_std": "exp(actor_logstd), the learned sampled-action standard deviation",
            "policy_update_l2": "L2 norm of actor parameter movement in the latest PPO update",
            "policy_update_rel_l2": "policy_update_l2 divided by pre-update actor parameter norm",
        },
        "available_actions": action_surface,
        "safety_bounds": {
            "actor_lr": {"min": 1e-5, "max": 3e-3, "current": recipe.actor_lr},
            "critic_lr": {"min": 1e-5, "max": 3e-3, "current": recipe.critic_lr},
            "entropy_coef": {"min": 0.0, "max": 0.1, "current": recipe.entropy_coef},
            "clip_coef": {"min": 0.03, "max": 0.3, "current": recipe.clip_coef},
            "update_epochs": {"min": 1, "max": 8, "current": recipe.update_epochs},
            "target_kl": {"min": 0.003, "max": 0.1, "current": recipe.target_kl},
            "collision_penalty": {"min": 1.0, "max": 50.0, "current": recipe.collision_penalty},
            "success_bonus": {"min": 2.5, "max": 10.0, "current": recipe.success_bonus},
            "progress_scale": {"min": 1.0, "max": 5.0, "current": recipe.progress_scale},
        },
        "run_state_constraints": {
            "stop_training_allowed": stop_allowed,
            "rollback_available": bool(best_checkpoint_metrics) and bool(rollback_gate.get("gate_passed", False)),
            "best_checkpoint_available": bool(best_checkpoint_metrics),
        },
        "rollback_gate": rollback_gate,
        "current_recipe": asdict(recipe),
        "latest_telemetry": {
            **latest,
            "update": update,
            "global_step": global_step,
            "deterministic_eval": deterministic_evals,
            "stochastic_eval": stochastic_evals,
            "deterministic_eval_summary": det_summary,
            "stochastic_eval_summary": stoch_summary,
            "drop_from_best_robust_score": best_robust_score - current_robust_score,
        },
        "recent_telemetry": telemetry_history[-8:],
        "best_checkpoint": {
            "available": bool(best_checkpoint_metrics),
            "metrics": best_checkpoint_metrics or {},
            "selection_criterion": CHECKPOINT_SELECTION_CRITERION,
            "robust_score": best_robust_score,
        },
        "decision_history": decision_history[-6:],
    }


def main() -> None:
    args = parse_args()
    if args.mode == "ai_manager" and args.anneal_lr:
        raise ValueError(
            "--anneal_lr cannot be combined with --mode ai_manager because it creates a second "
            "learning-rate controller outside the manager's current_recipe and action semantics"
        )
    seed_everything(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recipe = TrainingRecipe(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_coef=args.entropy_coef,
        clip_coef=args.clip_coef,
        update_epochs=args.update_epochs,
        target_kl=args.target_kl,
        collision_penalty=args.collision_penalty,
        success_bonus=args.success_bonus,
        progress_scale=args.progress_scale,
    )
    recipe.clamp()
    env_config = ReacherConfig(
        max_steps=args.max_episode_steps,
        success_radius=args.success_radius,
        collision_penalty=recipe.collision_penalty,
        success_bonus=recipe.success_bonus,
        progress_scale=recipe.progress_scale,
    )
    envs = [
        make_env(
            max_steps=args.max_episode_steps,
            success_radius=args.success_radius,
            collision_penalty=recipe.collision_penalty,
            success_bonus=recipe.success_bonus,
            progress_scale=recipe.progress_scale,
        )
        for _ in range(args.num_envs)
    ]
    obs_list = [env.reset(seed=args.seed * 10_000 + i)[0] for i, env in enumerate(envs)]
    obs_np = np.stack(obs_list).astype(np.float32)
    obs_dim = obs_np.shape[1]
    action_dim = int(np.prod(envs[0].action_space.shape))

    model = ActorCritic(obs_dim, action_dim, args.initial_actor_logstd).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": list(model.actor_mean.parameters()) + [model.actor_logstd], "lr": recipe.actor_lr},
            {"params": model.critic.parameters(), "lr": recipe.critic_lr},
        ],
        eps=1e-5,
    )
    diagnostic_eval_seed_base = args.seed * 1_000_000 + 100_000

    config = {
        **vars(args),
        "device": str(device),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "env_config": asdict(env_config),
        "initial_recipe": asdict(recipe),
        "task_note": "start pose, target, obstacle location, and obstacle radius are randomized every episode",
        "evaluation_protocol": {
            "training_diagnostic_eval": (
                "A fixed easy/medium/hard scene set evaluated repeatedly during training in both "
                "deterministic and stochastic modes. These diagnostics are visible to the AI manager "
                "and are used for checkpoint selection."
            ),
            "training_diagnostic_seed_base": diagnostic_eval_seed_base,
            "final_heldout_eval": (
                "Separate easy/medium/hard seed range evaluated only after training is complete. "
                "These episodes are never shown to the manager and do not affect checkpoint selection."
            ),
            "final_heldout_artifacts": [
                "final_heldout_eval.json",
                "final_heldout_eval_by_difficulty.csv",
            ],
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    metrics_path = out_dir / "metrics.csv"
    telemetry_path = out_dir / "telemetry.jsonl"
    updates_path = out_dir / "updates.jsonl"
    decisions_path = out_dir / "controller_decisions.jsonl"
    fieldnames = [
        "update",
        "global_step",
        "elapsed_seconds",
        "actor_lr",
        "critic_lr",
        "entropy_coef",
        "clip_coef",
        "update_epochs",
        "configured_update_epochs",
        "epochs_ran",
        "epoch_utilization",
        "target_kl",
        "kl_to_target",
        "kl_early_stopped",
        "collision_penalty",
        "success_bonus",
        "progress_scale",
        "actor_logstd",
        "actor_action_std",
        "sampled_action_saturation_fraction",
        "policy_entropy",
        "policy_update_l2",
        "policy_update_rel_l2",
        "approx_kl",
        "clip_fraction",
        "value_loss",
        "policy_loss",
        "explained_variance",
        "train_return_mean",
        "train_safe_success_rate",
        "train_collision_episode_rate",
        "eval_safe_success_mean",
        "eval_collision_mean",
        "eval_min_distance_mean",
        "eval_final_distance_mean",
        "eval_episode_length_mean",
        "eval_steps_to_success_mean",
        "eval_progress_reward_mean",
        "eval_collision_penalty_reward_mean",
        "eval_success_bonus_reward_mean",
        "stochastic_eval_safe_success_mean",
        "stochastic_eval_collision_mean",
        "checkpoint_robust_score",
        "checkpoint_mean_utility",
        "checkpoint_mean_safe_success",
        "checkpoint_worst_eval_cell_safe_success",
        "checkpoint_mean_collision",
        "checkpoint_worst_eval_cell_collision",
        "stochastic_eval_easy_safe_success",
        "stochastic_eval_medium_safe_success",
        "stochastic_eval_hard_safe_success",
        "stochastic_eval_easy_collision",
        "stochastic_eval_medium_collision",
        "stochastic_eval_hard_collision",
        "eval_easy_safe_success",
        "eval_medium_safe_success",
        "eval_hard_safe_success",
        "eval_easy_collision",
        "eval_medium_collision",
        "eval_hard_collision",
        "eval_easy_min_distance",
        "eval_medium_min_distance",
        "eval_hard_min_distance",
        "eval_easy_final_distance",
        "eval_medium_final_distance",
        "eval_hard_final_distance",
        "eval_easy_episode_length",
        "eval_medium_episode_length",
        "eval_hard_episode_length",
        "eval_easy_steps_to_success",
        "eval_medium_steps_to_success",
        "eval_hard_steps_to_success",
        "eval_easy_progress_reward",
        "eval_medium_progress_reward",
        "eval_hard_progress_reward",
        "eval_easy_collision_penalty_reward",
        "eval_medium_collision_penalty_reward",
        "eval_hard_collision_penalty_reward",
        "eval_easy_success_bonus_reward",
        "eval_medium_success_bonus_reward",
        "eval_hard_success_bonus_reward",
        "pre_manager_eval_safe_success_mean",
        "pre_manager_eval_collision_mean",
        "pre_manager_stochastic_eval_safe_success_mean",
        "post_manager_eval_safe_success_mean",
        "post_manager_eval_collision_mean",
        "post_manager_stochastic_eval_safe_success_mean",
        "best_checkpoint_robust_score",
        "drop_from_best_robust_score",
        "rollback_trigger_metric",
        "best_checkpoint_deterministic_safe_success",
        "diagnostic_only_deterministic_safe_success_gap_from_best",
        "rollback_eligible_streak",
        "rollback_cooldown_active",
        "rollback_gate_passed",
        "manager_action",
        "manager_wait_seconds",
        "manager_wait_seconds_total",
    ]
    write_metrics_header(metrics_path, fieldnames)

    recent_returns: deque[float] = deque(maxlen=200)
    recent_safe: deque[float] = deque(maxlen=200)
    recent_collision: deque[float] = deque(maxlen=200)
    current_returns = np.zeros(args.num_envs, dtype=np.float32)
    current_safe = np.zeros(args.num_envs, dtype=np.float32)
    current_collision = np.zeros(args.num_envs, dtype=np.float32)
    next_done_np = np.zeros(args.num_envs, dtype=np.float32)

    global_step = 0
    start_time = time.time()
    best_checkpoint_key: tuple[float, ...] | None = None
    best_checkpoint_metrics: dict[str, Any] | None = None
    rollback_eligible_streak = 0
    last_rollback_update = -10**9
    telemetry_history: list[dict[str, Any]] = []
    decision_history: list[dict[str, Any]] = []
    manager = (
        RandomizedReacherManager(
            model=args.llm_model,
            temperature=args.llm_temperature,
            formatter_model=args.llm_formatter_model,
            formatter_temperature=args.llm_formatter_temperature,
        )
        if args.mode == "ai_manager"
        else None
    )
    batch_size = args.num_envs * args.steps_per_update
    run_plan = {
        "total_updates": args.total_updates,
        "total_timesteps": args.total_updates * batch_size,
        "eval_interval_updates": args.eval_interval,
        "eval_episodes_per_split": args.eval_episodes,
        "manager_application_mode": "async",
        "async_lead_updates": args.async_manager_lead_updates,
        "async_max_wait_updates": args.async_manager_max_wait_updates,
        "async_wait_timeout_seconds": args.async_manager_wait_timeout_seconds,
    }
    manager_executor = ThreadPoolExecutor(max_workers=1) if manager is not None else None
    pending_manager_call: dict[str, Any] | None = None
    completed_update = 0
    manager_wait_seconds_total = 0.0
    manager_wait_events = 0

    def apply_manager_decision(
        *,
        decision: dict[str, Any],
        context: dict[str, Any],
        row: dict[str, Any],
        evals: dict[str, dict[str, float]],
        stochastic_evals: dict[str, dict[str, float]],
        update: int,
        global_step: int,
    ) -> tuple[bool, dict[str, dict[str, float]], dict[str, dict[str, float]], str]:
        nonlocal recipe
        nonlocal best_checkpoint_metrics
        nonlocal last_rollback_update
        nonlocal rollback_eligible_streak

        selected = decision.get("selected_action", {})
        action_name = selected.get("name", "no_action")
        policy_update = selected.get("policy_update", {}) or {}
        action_label = str(action_name).upper()
        rollback_gate = context.get("rollback_gate", {})
        rollback_gate_passed = bool(rollback_gate.get("gate_passed", False))
        if "rollback_gate_passed" in row:
            rollback_gate_passed = bool(row["rollback_gate_passed"])
        rollback_requested = action_name in {"rollback_to_best", "rollback_and_consolidate"}
        rollback_applied = False
        telemetry_before = dict(row)

        if rollback_requested and not rollback_gate_passed:
            action_label = "ROLLBACK_GATE_REJECTED"
            decision["rollback_gate_rejection"] = (
                "rollback was no longer eligible under fresh application-time telemetry"
            )
            decision["selected_action"]["policy_update"] = {}
            policy_update = {}
            action_name = "no_action"

        if action_name == "stop_training":
            fresh_stop_allowed = meets_robust_stop_criterion(row)
            if not fresh_stop_allowed:
                action_label = "STOP_GATE_REJECTED"
                decision["stop_gate_rejection"] = (
                    "stop_training was no longer eligible under fresh application-time telemetry"
                )
                decision["selected_action"]["policy_update"] = {}
                policy_update = {}
                action_name = "no_action"

        if action_name == "rollback_to_best" and (out_dir / "best.pt").exists():
            recipe, checkpoint = load_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                device,
                recipe_override=recipe,
            )
            apply_reward_recipe_to_envs(envs, recipe)
            best_checkpoint_metrics = checkpoint.get("metrics", best_checkpoint_metrics)
            action_label = "ROLLBACK_TO_BEST"
            rollback_applied = True
        elif action_name == "rollback_and_consolidate" and (out_dir / "best.pt").exists():
            recipe, checkpoint = load_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                device,
                recipe_override=recipe,
            )
            recipe = apply_recipe_update(recipe, policy_update)
            set_optimizer_learning_rates(optimizer, recipe)
            apply_reward_recipe_to_envs(envs, recipe)
            best_checkpoint_metrics = checkpoint.get("metrics", best_checkpoint_metrics)
            action_label = "ROLLBACK_AND_CONSOLIDATE"
            rollback_applied = True
        elif action_name == "stop_training":
            action_label = "STOP_TRAINING"
        else:
            recipe = apply_recipe_update(recipe, policy_update)
            apply_reward_recipe_to_envs(envs, recipe)
            if policy_update:
                action_label = f"{str(action_name).upper()}_UPDATE"

        if rollback_applied:
            last_rollback_update = update
            rollback_eligible_streak = 0
            evals, stochastic_evals = run_eval_suite(
                model,
                device,
                eval_episodes=args.eval_episodes,
                seed_base=diagnostic_eval_seed_base,
                max_episode_steps=args.max_episode_steps,
                success_radius=args.success_radius,
                collision_penalty=recipe.collision_penalty,
                success_bonus=recipe.success_bonus,
                progress_scale=recipe.progress_scale,
            )
            update_row_from_evals(row, evals, stochastic_evals)
            row["post_manager_eval_safe_success_mean"] = row["eval_safe_success_mean"]
            row["post_manager_eval_collision_mean"] = row["eval_collision_mean"]
            row["post_manager_stochastic_eval_safe_success_mean"] = row[
                "stochastic_eval_safe_success_mean"
            ]
            row["rollback_eligible_streak"] = rollback_eligible_streak
            row["rollback_cooldown_active"] = True
            row["rollback_gate_passed"] = False

        row["manager_action"] = action_label
        decision_record = {
            "update": update,
            "global_step": global_step,
            "action": action_label,
            "application_mode": "async",
            "decision_context_update": context.get("latest_telemetry", {}).get("update"),
            "application_lag_updates": update
            - int(context.get("latest_telemetry", {}).get("update", update)),
            "recipe_after": asdict(recipe),
            "context": context,
            "row_after_manager": row,
            **decision,
        }
        append_jsonl(decisions_path, decision_record)
        decision_history.append(
            {
                "decision_update": update,
                "request_context_update": context.get("latest_telemetry", {}).get("update"),
                "application_update": update,
                "application_lag_updates": update
                - int(context.get("latest_telemetry", {}).get("update", update)),
                "action": action_label,
                "selected_control": str(action_name),
                "selected_policy": dict(policy_update),
                "rationale_then": decision.get("reason"),
                "expected_outcome": selected.get("expected_outcome"),
                "telemetry_before": telemetry_before,
                "recipe_after": asdict(recipe),
            }
        )
        return action_name == "stop_training", evals, stochastic_evals, action_label

    stop_requested = False
    for update in range(1, args.total_updates + 1):
        completed_update = update
        frac = 1.0 - (update - 1.0) / args.total_updates
        actor_lr = recipe.actor_lr * frac if args.anneal_lr else recipe.actor_lr
        critic_lr = recipe.critic_lr * frac if args.anneal_lr else recipe.critic_lr
        optimizer.param_groups[0]["lr"] = actor_lr
        optimizer.param_groups[1]["lr"] = critic_lr

        obs_buf = torch.zeros((args.steps_per_update, args.num_envs, obs_dim), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((args.steps_per_update, args.num_envs, action_dim), dtype=torch.float32, device=device)
        logprobs_buf = torch.zeros((args.steps_per_update, args.num_envs), dtype=torch.float32, device=device)
        rewards_buf = torch.zeros((args.steps_per_update, args.num_envs), dtype=torch.float32, device=device)
        dones_buf = torch.zeros((args.steps_per_update, args.num_envs), dtype=torch.float32, device=device)
        values_buf = torch.zeros((args.steps_per_update, args.num_envs), dtype=torch.float32, device=device)

        for step in range(args.steps_per_update):
            global_step += args.num_envs
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            obs_buf[step] = obs_t
            dones_buf[step] = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_t, logprob_t, _, value_t = model.get_action_and_value(obs_t)
            actions_buf[step] = action_t
            logprobs_buf[step] = logprob_t
            values_buf[step] = value_t

            action_np = action_t.cpu().numpy()
            next_obs = []
            rewards = []
            dones = []
            for i, env in enumerate(envs):
                obs_next, reward, terminated, truncated, info = env.step(action_np[i])
                done = bool(terminated or truncated)
                current_returns[i] += float(reward)
                current_safe[i] = max(current_safe[i], float(info["safe_success"]))
                current_collision[i] = max(current_collision[i], float(info["is_colliding"]))
                if done:
                    recent_returns.append(float(current_returns[i]))
                    recent_safe.append(float(current_safe[i]))
                    recent_collision.append(float(current_collision[i]))
                    current_returns[i] = 0.0
                    current_safe[i] = 0.0
                    current_collision[i] = 0.0
                    obs_next, _ = env.reset()
                next_obs.append(obs_next)
                rewards.append(float(reward))
                dones.append(float(done))
            rewards_buf[step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            next_done_np = np.asarray(dones, dtype=np.float32)
            obs_np = np.stack(next_obs).astype(np.float32)

        with torch.no_grad():
            next_value = model.get_value(torch.as_tensor(obs_np, dtype=torch.float32, device=device))
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
            for t in reversed(range(args.steps_per_update)):
                if t == args.steps_per_update - 1:
                    next_nonterminal = 1.0 - torch.as_tensor(next_done_np, dtype=torch.float32, device=device)
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones_buf[t + 1]
                    next_values = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * next_values * next_nonterminal - values_buf[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = actions_buf.reshape((-1, action_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        indices = np.arange(batch_size)
        approx_kl = torch.tensor(0.0, device=device)
        clip_fraction = 0.0
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        entropy = torch.tensor(0.0, device=device)
        epochs_ran = 0
        actor_params_before = actor_parameter_vector(model).clone()
        for _ in range(recipe.update_epochs):
            epochs_ran += 1
            np.random.shuffle(indices)
            for start in range(0, batch_size, args.minibatch_size):
                mb_idx = torch.as_tensor(indices[start : start + args.minibatch_size], dtype=torch.long, device=device)
                _, new_logprob, entropy, new_value = model.get_action_and_value(b_obs[mb_idx], b_actions[mb_idx])
                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_fraction = float(((ratio - 1.0).abs() > recipe.clip_coef).float().mean().cpu())
                pg_loss1 = -b_advantages[mb_idx] * ratio
                pg_loss2 = -b_advantages[mb_idx] * torch.clamp(ratio, 1 - recipe.clip_coef, 1 + recipe.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()
                loss = policy_loss - recipe.entropy_coef * entropy.mean() + args.vf_coef * value_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            if recipe.target_kl > 0 and float(approx_kl.detach().cpu()) > recipe.target_kl:
                break

        actor_params_after = actor_parameter_vector(model)
        policy_update_l2 = float(torch.linalg.vector_norm(actor_params_after - actor_params_before).cpu())
        actor_param_l2 = float(torch.linalg.vector_norm(actor_params_before).cpu())
        policy_update_rel_l2 = policy_update_l2 / max(actor_param_l2, 1e-12)
        explained = explained_variance(b_values, b_returns)
        configured_update_epochs = int(recipe.update_epochs)
        approx_kl_value = float(approx_kl.detach().cpu())
        actor_logstd_mean = float(model.actor_logstd.detach().mean().cpu())
        sampled_action_saturation_fraction = float((actions_buf.abs() > 1.0).float().mean().cpu())
        epoch_utilization = float(epochs_ran / max(configured_update_epochs, 1))
        update_payload = {
            "update": update,
            "global_step": global_step,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "entropy_coef": recipe.entropy_coef,
            "clip_coef": recipe.clip_coef,
            "update_epochs": configured_update_epochs,
            "configured_update_epochs": configured_update_epochs,
            "epochs_ran": epochs_ran,
            "epoch_utilization": epoch_utilization,
            "target_kl": recipe.target_kl,
            "kl_to_target": approx_kl_value / max(float(recipe.target_kl), 1e-12),
            "kl_early_stopped": bool(epochs_ran < configured_update_epochs),
            "collision_penalty": recipe.collision_penalty,
            "success_bonus": recipe.success_bonus,
            "progress_scale": recipe.progress_scale,
            "actor_logstd": actor_logstd_mean,
            "actor_action_std": float(math.exp(actor_logstd_mean)),
            "sampled_action_saturation_fraction": sampled_action_saturation_fraction,
            "policy_entropy": float(entropy.mean().detach().cpu()),
            "policy_update_l2": policy_update_l2,
            "policy_update_rel_l2": policy_update_rel_l2,
            "approx_kl": approx_kl_value,
            "clip_fraction": clip_fraction,
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "explained_variance": explained,
            "train_return_mean": float(np.mean(recent_returns)) if recent_returns else 0.0,
            "train_safe_success_rate": float(np.mean(recent_safe)) if recent_safe else 0.0,
            "train_collision_episode_rate": float(np.mean(recent_collision)) if recent_collision else 0.0,
            "train_window_completed_episodes": len(recent_returns),
            "elapsed_seconds": time.time() - start_time,
        }
        append_jsonl(updates_path, update_payload)

        async_action_label_for_row: str | None = None
        manager_wait_seconds = 0.0
        ready_async_call: dict[str, Any] | None = None
        if pending_manager_call is not None:
            request_update = int(pending_manager_call["request_update"])
            target_update = int(pending_manager_call["target_update"])
            age = update - request_update
            future: Future[dict[str, Any]] = pending_manager_call["future"]
            if update >= target_update and future.done():
                ready_async_call = pending_manager_call
                pending_manager_call = None
            elif age >= args.async_manager_max_wait_updates:
                # The decision has taken too long to arrive on its own. Rather than
                # discard work the manager has already done, stop training here and
                # block until it lands, so no manager response is ever wasted.
                wait_started = time.time()
                try:
                    future.result(timeout=args.async_manager_wait_timeout_seconds)
                    timed_out = False
                except FuturesTimeoutError:
                    timed_out = True
                except Exception:  # noqa: BLE001 - surfaced through the decision log below.
                    timed_out = False
                manager_wait_seconds = time.time() - wait_started
                manager_wait_seconds_total += manager_wait_seconds
                manager_wait_events += 1
                if timed_out:
                    append_jsonl(
                        decisions_path,
                        {
                            "update": update,
                            "global_step": global_step,
                            "action": "ASYNC_MANAGER_STALE",
                            "application_mode": "async",
                            "request_update": request_update,
                            "target_update": target_update,
                            "age_updates": age,
                            "done": future.done(),
                            "waited_seconds": manager_wait_seconds,
                            "wait_timeout_seconds": args.async_manager_wait_timeout_seconds,
                            "recipe_after": asdict(recipe),
                        },
                    )
                    pending_manager_call = None
                    async_action_label_for_row = "ASYNC_MANAGER_STALE"
                else:
                    append_jsonl(
                        decisions_path,
                        {
                            "update": update,
                            "global_step": global_step,
                            "action": "ASYNC_MANAGER_WAITED",
                            "application_mode": "async",
                            "request_update": request_update,
                            "target_update": target_update,
                            "age_updates": age,
                            "waited_seconds": manager_wait_seconds,
                            "recipe_after": asdict(recipe),
                        },
                    )
                    print(
                        f"update={update:04d} step={global_step:07d} "
                        f"async_manager_wait={manager_wait_seconds:.1f}s age_updates={age}",
                        flush=True,
                    )
                    ready_async_call = pending_manager_call
                    pending_manager_call = None

        manager_request_offset = args.eval_interval - args.async_manager_lead_updates
        is_async_request_update = (
            manager is not None
            and manager_request_offset > 0
            and update > 1
            and update % args.eval_interval == manager_request_offset
        )
        is_eval_update = update % args.eval_interval == 0 or update == 1 or update == args.total_updates
        if is_eval_update or is_async_request_update or ready_async_call is not None:
            evals, stochastic_evals = run_eval_suite(
                model,
                device,
                eval_episodes=args.eval_episodes,
                seed_base=diagnostic_eval_seed_base,
                max_episode_steps=args.max_episode_steps,
                success_radius=args.success_radius,
                collision_penalty=recipe.collision_penalty,
                success_bonus=recipe.success_bonus,
                progress_scale=recipe.progress_scale,
            )
            row = {
                **update_payload,
                "pre_manager_eval_safe_success_mean": "",
                "pre_manager_eval_collision_mean": "",
                "pre_manager_stochastic_eval_safe_success_mean": "",
                "post_manager_eval_safe_success_mean": "",
                "post_manager_eval_collision_mean": "",
                "post_manager_stochastic_eval_safe_success_mean": "",
                "manager_action": async_action_label_for_row or "KEEP",
                "manager_wait_seconds": manager_wait_seconds,
                "manager_wait_seconds_total": manager_wait_seconds_total,
            }
            update_row_from_evals(row, evals, stochastic_evals)
            current_checkpoint_key = checkpoint_selection_key(row)
            if best_checkpoint_key is None or current_checkpoint_key > best_checkpoint_key:
                best_checkpoint_key = current_checkpoint_key
                best_checkpoint_metrics = dict(row)
                save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    optimizer,
                    recipe,
                    config,
                    update,
                    global_step,
                    row,
                )

            best_update = int(best_checkpoint_metrics["update"]) if best_checkpoint_metrics else None
            best_safe = float((best_checkpoint_metrics or {}).get("eval_safe_success_mean", 0.0))
            best_robust_score = float((best_checkpoint_metrics or {}).get("checkpoint_robust_score", 0.0))
            current_robust_score = float(row["checkpoint_robust_score"])
            drop_from_best_robust = max(0.0, best_robust_score - current_robust_score)
            drop_from_best_safe = max(0.0, best_safe - float(row["eval_safe_success_mean"]))
            can_compare_to_prior_best = best_update is not None and best_update < update
            if can_compare_to_prior_best and drop_from_best_robust >= args.rollback_min_drop:
                rollback_eligible_streak += 1
            else:
                rollback_eligible_streak = 0
            cooldown_updates = args.eval_interval * args.rollback_cooldown_intervals
            rollback_cooldown_active = (update - last_rollback_update) < cooldown_updates
            rollback_gate_passed = (
                can_compare_to_prior_best
                and rollback_eligible_streak >= args.rollback_patience
                and not rollback_cooldown_active
            )
            rollback_gate = {
                # The gate triggers on the robust score, which spans deterministic and
                # stochastic evaluation. The deterministic-only figures below are
                # diagnostic context and are deliberately not part of the trigger.
                "trigger_metric": "drop_from_best_robust_score",
                "selection_criterion": CHECKPOINT_SELECTION_CRITERION,
                "best_robust_score": best_robust_score,
                "current_robust_score": current_robust_score,
                "drop_from_best_robust_score": drop_from_best_robust,
                "min_drop": args.rollback_min_drop,
                "patience_evals": args.rollback_patience,
                "eligible_streak": rollback_eligible_streak,
                "cooldown_intervals": args.rollback_cooldown_intervals,
                "cooldown_updates": cooldown_updates,
                "cooldown_active": rollback_cooldown_active,
                "gate_passed": rollback_gate_passed,
                "best_update": best_update,
                "diagnostic_only_best_checkpoint_deterministic_safe_success": best_safe,
                "diagnostic_only_current_deterministic_safe_success": row["eval_safe_success_mean"],
            }
            row.update(
                {
                    "best_checkpoint_robust_score": best_robust_score,
                    "drop_from_best_robust_score": drop_from_best_robust,
                    "rollback_trigger_metric": "drop_from_best_robust_score",
                    "best_checkpoint_deterministic_safe_success": best_safe,
                    "diagnostic_only_deterministic_safe_success_gap_from_best": drop_from_best_safe,
                    "rollback_eligible_streak": rollback_eligible_streak,
                    "rollback_cooldown_active": rollback_cooldown_active,
                    "rollback_gate_passed": rollback_gate_passed,
                }
            )
            if ready_async_call is not None:
                row["pre_manager_eval_safe_success_mean"] = row["eval_safe_success_mean"]
                row["pre_manager_eval_collision_mean"] = row["eval_collision_mean"]
                row["pre_manager_stochastic_eval_safe_success_mean"] = row[
                    "stochastic_eval_safe_success_mean"
                ]
                stop_requested, evals, stochastic_evals, async_action_label_for_row = apply_manager_decision(
                    decision=ready_async_call["future"].result(),
                    context=ready_async_call["context"],
                    row=row,
                    evals=evals,
                    stochastic_evals=stochastic_evals,
                    update=update,
                    global_step=global_step,
                )
                print(
                    f"update={update:04d} step={global_step:07d} "
                    f"async_manager_applied={async_action_label_for_row}",
                    flush=True,
                )

            if manager is not None and is_async_request_update:
                row["pre_manager_eval_safe_success_mean"] = row["eval_safe_success_mean"]
                row["pre_manager_eval_collision_mean"] = row["eval_collision_mean"]
                row["pre_manager_stochastic_eval_safe_success_mean"] = row[
                    "stochastic_eval_safe_success_mean"
                ]
                context = build_manager_context(
                    update=update,
                    global_step=global_step,
                    recipe=recipe,
                    latest=row,
                    deterministic_evals=evals,
                    stochastic_evals=stochastic_evals,
                    telemetry_history=telemetry_history,
                    decision_history=decision_history,
                    best_checkpoint_metrics=best_checkpoint_metrics,
                    rollback_gate=rollback_gate,
                    run_plan=run_plan,
                )
                if pending_manager_call is None and manager_executor is not None:
                    pending_manager_call = {
                        "future": manager_executor.submit(manager.decide, context),
                        "context": context,
                        "evals": evals,
                        "stochastic_evals": stochastic_evals,
                        "request_update": update,
                        "target_update": update + args.async_manager_lead_updates,
                    }
                    row["manager_action"] = "ASYNC_MANAGER_REQUESTED"
                    append_jsonl(
                        decisions_path,
                        {
                            "update": update,
                            "global_step": global_step,
                            "action": "ASYNC_MANAGER_REQUESTED",
                            "application_mode": "async",
                            "target_update": update + args.async_manager_lead_updates,
                            "recipe_after": asdict(recipe),
                            "context": context,
                        },
                    )
                else:
                    row["manager_action"] = "ASYNC_MANAGER_REQUEST_SKIPPED_PENDING"

            append_metrics_row(metrics_path, fieldnames, row)
            append_jsonl(telemetry_path, {**row, "eval": evals, "stochastic_eval": stochastic_evals})
            telemetry_history.append(row)
            print(
                f"update={update:04d} step={global_step:07d} "
                f"eval_safe={row['eval_safe_success_mean']:.3f} "
                f"collision={row['eval_collision_mean']:.3f} "
                f"train_safe={row['train_safe_success_rate']:.3f} "
                f"manager={row['manager_action']}",
                flush=True,
            )
            if stop_requested:
                break

    if pending_manager_call is not None:
        # Training finished before this decision landed. There is no interval left to
        # apply it to, so it is recorded as abandoned rather than silently dropped.
        append_jsonl(
            decisions_path,
            {
                "update": completed_update,
                "global_step": global_step,
                "action": "ASYNC_MANAGER_ABANDONED_AT_END",
                "application_mode": "async",
                "request_update": pending_manager_call["request_update"],
                "target_update": pending_manager_call["target_update"],
                "recipe_after": asdict(recipe),
            },
        )

    if manager is not None:
        total_elapsed = time.time() - start_time
        wait_summary = {
            "manager_wait_seconds_total": manager_wait_seconds_total,
            "manager_wait_events": manager_wait_events,
            "mean_manager_wait_seconds": (
                manager_wait_seconds_total / manager_wait_events if manager_wait_events else 0.0
            ),
            "total_elapsed_seconds": total_elapsed,
            "fraction_of_run_spent_waiting_on_manager": (
                manager_wait_seconds_total / total_elapsed if total_elapsed > 0 else 0.0
            ),
            "max_wait_updates": args.async_manager_max_wait_updates,
            "wait_timeout_seconds": args.async_manager_wait_timeout_seconds,
        }
        (out_dir / "manager_wait_summary.json").write_text(json.dumps(wait_summary, indent=2) + "\n")
        print(
            f"manager wait: {manager_wait_seconds_total:.1f}s across {manager_wait_events} pause(s) "
            f"({100.0 * wait_summary['fraction_of_run_spent_waiting_on_manager']:.1f}% of wall clock)",
            flush=True,
        )

    if manager_executor is not None:
        manager_executor.shutdown(wait=False, cancel_futures=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "recipe": asdict(recipe),
            "update": completed_update,
            "global_step": global_step,
        },
        out_dir / "final.pt",
    )

    if not args.skip_final_eval:
        write_final_heldout_eval(
            out_dir,
            model,
            optimizer,
            device,
            recipe=recipe,
            config=config,
            completed_update=completed_update,
            global_step=global_step,
            best_checkpoint_path=out_dir / "best.pt",
            best_checkpoint_metrics=best_checkpoint_metrics,
            final_eval_episodes=args.final_eval_episodes,
            final_eval_seed_base=args.seed * 10_000_000 + args.final_eval_seed_base,
            max_episode_steps=args.max_episode_steps,
            success_radius=args.success_radius,
        )


if __name__ == "__main__":
    main()
