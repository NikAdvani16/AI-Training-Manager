from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from manager import TrainingRecipe, allowed_actions, build_analysis_prompt, build_prompt
from run_experiment import CONFIG_DIR, ROOT, load_json


def example_context(condition_name: str, seed: int) -> dict:
    condition = load_json(CONFIG_DIR / f"{condition_name}.json")
    common = load_json(CONFIG_DIR / "common.json")
    recipe_data = dict(condition["recipe"])
    recipe_data.pop("initial_actor_logstd")
    recipe = TrainingRecipe(**recipe_data)
    update = 15
    global_step = update * common["protocol"]["num_envs"] * common["protocol"]["steps_per_update"]
    split = {
        "easy": {"safe_success_rate": 0.44, "collision_episode_rate": 0.12, "mean_final_distance": 0.25, "mean_min_distance": 0.13},
        "medium": {"safe_success_rate": 0.31, "collision_episode_rate": 0.21, "mean_final_distance": 0.34, "mean_min_distance": 0.19},
        "hard": {"safe_success_rate": 0.18, "collision_episode_rate": 0.30, "mean_final_distance": 0.42, "mean_min_distance": 0.25},
    }
    stochastic = {
        "easy": {"safe_success_rate": 0.38, "collision_episode_rate": 0.17, "mean_final_distance": 0.28, "mean_min_distance": 0.15},
        "medium": {"safe_success_rate": 0.27, "collision_episode_rate": 0.25, "mean_final_distance": 0.36, "mean_min_distance": 0.21},
        "hard": {"safe_success_rate": 0.14, "collision_episode_rate": 0.34, "mean_final_distance": 0.45, "mean_min_distance": 0.27},
    }
    latest = {
        "update": update,
        "global_step": global_step,
        "train_safe_success_rate": 0.29,
        "train_collision_episode_rate": 0.24,
        "train_window_completed_episodes": 200,
        "deterministic_eval": split,
        "deterministic_eval_summary": {"safe_success_mean": 0.31, "collision_mean": 0.21, "final_distance_mean": 0.34, "min_distance_mean": 0.19},
        "stochastic_eval": stochastic,
        "stochastic_eval_summary": {"safe_success_mean": 0.2633, "collision_mean": 0.2533, "final_distance_mean": 0.3633, "min_distance_mean": 0.21},
        "approx_kl": 0.019,
        "kl_to_target": 0.6333,
        "clip_fraction": 0.11,
        "epochs_ran": recipe.update_epochs,
        "explained_variance": 0.42,
        "actor_action_std": 0.48,
        "sampled_action_saturation_fraction": 0.08,
        "policy_update_rel_l2": 0.014,
    }
    return {
        "available_actions": allowed_actions(recipe, allow_rollback=False, allow_stop=False),
        "current_recipe": asdict(recipe),
        "run_state_constraints": {"stop_training_allowed": False, "rollback_available": False},
        "latest_telemetry": latest,
        "recent_telemetry": [],
        "best_checkpoint": {"available": False, "metrics": {}},
        "decision_history": [],
        "rollback_gate": {},
        "run_plan": {
            "total_updates": common["protocol"]["total_updates"],
            "total_timesteps": common["protocol"]["total_updates"] * common["protocol"]["num_envs"] * common["protocol"]["steps_per_update"],
            "eval_episodes_per_split": common["protocol"]["eval_episodes"],
            "manager_application_mode": "async",
            "async_lead_updates": common["manager"]["async_lead_updates"],
            "async_stale_updates": common["manager"]["async_stale_updates"],
            "seed": seed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export exact rendered manager prompts without calling an API.")
    parser.add_argument("--condition", choices=["baseline", "conservative", "aggressive"], default="aggressive")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prompts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = example_context(args.condition, args.seed)
    expert_analysis = (
        "Example expert analysis used only to render the Stage 2 formatter prompt.\n\n"
        "Selected action name: no_action\n"
        "Exact policy object: {}\n"
        "Reason: This is a prompt-export example, not a live training decision.\n"
        "Expected next-interval outcome: Training continues with the current recipe.\n"
        "Success criteria: The next diagnostic evaluation supplies another comparable observation.\n"
        "Metrics to watch next: deterministic and stochastic safe success and collision rate"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1 = build_analysis_prompt(context)
    stage2 = build_prompt(context, validation_feedback=None, expert_analysis=expert_analysis)
    (args.output_dir / "stage1_example.txt").write_text(stage1, encoding="utf-8")
    (args.output_dir / "stage2_example.txt").write_text(stage2, encoding="utf-8")
    print(f"Wrote {args.output_dir / 'stage1_example.txt'}")
    print(f"Wrote {args.output_dir / 'stage2_example.txt'}")


if __name__ == "__main__":
    main()
