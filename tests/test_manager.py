from __future__ import annotations

import itertools
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from manager import (
    ACTION_DESCRIPTIONS,
    ACTION_NAMES,
    RandomizedReacherManager,
    TrainingRecipe,
    allowed_actions,
    apply_recipe_update,
    build_analysis_prompt,
    build_manager_observation,
    build_prompt,
    validate_response,
)
from train_ppo import (
    ActorCritic,
    checkpoint_robust_metrics,
    checkpoint_selection_key,
    load_checkpoint,
    meets_robust_stop_criterion,
)


def manager_context() -> dict:
    return {
        "task": {"private_marker": "TASK_MUST_NOT_REACH_DYNAMIC_CONTEXT"},
        "available_actions": {
            "no_action": {},
            "actor_lr": {"actor_lr_multiplier": [0.5]},
            "entropy": {"entropy_coef_multiplier": [0.5]},
            "combined_policy": {
                "two_or_three_of": {
                    "actor_lr_multiplier": [0.5],
                    "entropy_coef_multiplier": [0.5],
                }
            },
        },
        "current_recipe": {
            "actor_lr": 3e-4,
            "critic_lr": 1e-3,
            "entropy_coef": 0.01,
            "clip_coef": 0.2,
            "update_epochs": 6,
            "target_kl": 0.03,
            "collision_penalty": 5.0,
            "success_bonus": 5.0,
            "progress_scale": 3.0,
        },
        "run_state_constraints": {
            "stop_training_allowed": False,
            "rollback_available": False,
        },
        "latest_telemetry": {
            "private_marker": "UNSELECTED_TRAINER_FIELD",
            "update": 20,
            "global_step": 40960,
            "train_safe_success_rate": 0.25,
            "eval_safe_success_mean": 0.321,
            "eval_collision_mean": 0.123,
            "deterministic_eval": {
                "easy": {"safe_success_rate": 0.4, "collision_episode_rate": 0.1}
            },
            "deterministic_eval_summary": {
                "safe_success_mean": 0.321,
                "collision_mean": 0.123,
            },
            "stochastic_eval": {
                "easy": {"safe_success_rate": 0.3, "collision_episode_rate": 0.2}
            },
            "stochastic_eval_summary": {
                "safe_success_mean": 0.2,
                "collision_mean": 0.25,
            },
        },
        "recent_telemetry": [],
        "best_checkpoint": {"available": False, "metrics": {}},
        "decision_history": [],
        "rollback_gate": {},
        "safety_bounds": {"private_marker": "BOUNDS_MUST_NOT_REACH_FORMATTER"},
    }


def no_action_payload() -> dict:
    return {
        "selected_action": {
            "name": "no_action",
            "policy": {},
            "expected_outcome": "Continue unchanged for one interval.",
            "success_criteria": "The next evaluation clarifies the trend.",
        },
        "reason": "The expert selected no_action.",
        "watch_next": ["eval_safe_success_mean"],
    }


class FormatterPromptTests(unittest.TestCase):
    def test_analysis_prompt_separates_recipe_actions_and_dynamic_context(self) -> None:
        prompt = build_analysis_prompt(manager_context())

        self.assertEqual(prompt.count("actor_lr_multiplier: 0.5"), 1)
        self.assertEqual(prompt.count("critic_lr           0.001"), 1)
        self.assertIn("deterministic  overall         0.321", prompt)
        self.assertNotIn("UNSELECTED_TRAINER_FIELD", prompt)
        self.assertNotIn("BOUNDS_MUST_NOT_REACH_FORMATTER", prompt)
        self.assertNotIn("TASK_MUST_NOT_REACH_DYNAMIC_CONTEXT", prompt)
        self.assertIn("combined_policy  (choose two or three of:", prompt)

    def test_every_offered_action_is_described(self) -> None:
        prompt = build_analysis_prompt(manager_context())
        flattened = " ".join(prompt.split())

        for action in ("no_action", "actor_lr", "entropy", "combined_policy"):
            self.assertIn(action, prompt)
            self.assertIn(" ".join(ACTION_DESCRIPTIONS[action].split()), flattened)

    def test_every_runtime_action_has_a_description(self) -> None:
        self.assertEqual(ACTION_NAMES - set(ACTION_DESCRIPTIONS), set())

    def test_unavailable_actions_are_not_mentioned_at_all(self) -> None:
        context = manager_context()
        context["rollback_gate"] = {
            "best_update": 10,
            "min_drop": 0.1,
            "patience_evals": 2,
            "eligible_streak": 0,
            "cooldown_active": False,
            "drop_from_best_robust_score": 0.02,
        }

        prompt = build_analysis_prompt(context)

        self.assertNotIn("rollback_to_best", prompt)
        self.assertNotIn("rollback_and_consolidate", prompt)
        self.assertNotIn("stop_training", prompt)
        self.assertNotIn("Not available", prompt)

    def test_run_position_and_eval_sample_size_are_stated(self) -> None:
        context = manager_context()
        context["run_plan"] = {
            "total_updates": 500,
            "total_timesteps": 1_024_000,
            "eval_episodes_per_split": 80,
        }

        prompt = build_analysis_prompt(context)

        self.assertIn("update 20 of 500", prompt)
        self.assertIn("global step 40960 of 1024000", prompt)
        self.assertIn("80 episodes per split", prompt)

    def test_async_timing_is_stated_without_claiming_training_waits(self) -> None:
        context = manager_context()
        context["run_plan"] = {
            "manager_application_mode": "async",
            "async_lead_updates": 5,
            "async_max_wait_updates": 10,
            "async_wait_timeout_seconds": 600,
        }

        prompt = build_analysis_prompt(context)

        self.assertIn("Training continues while you reason", prompt)
        self.assertIn("about 5 PPO updates", prompt)
        self.assertIn("by 10 updates after the snapshot", prompt)
        self.assertIn("at most 600 seconds", prompt)

    def test_action_space_is_described_without_judging_values(self) -> None:
        prompt = build_analysis_prompt(manager_context())

        self.assertIn("two-link robot arm", prompt)
        self.assertIn("clipped to the range [-1, 1]", prompt)
        for judgement in ("insane", "too high", "too low", "abnormal", "typical range"):
            self.assertNotIn(judgement, prompt)

    def test_rollback_and_consolidate_lists_its_exact_values(self) -> None:
        context = manager_context()
        context["available_actions"]["rollback_and_consolidate"] = {
            "exactly_one_of": {
                "actor_lr_multiplier": [0.5, 0.75],
                "clip_coef": [0.08, 0.12],
            }
        }

        prompt = build_analysis_prompt(context)

        self.assertIn("rollback_and_consolidate  choose exactly one of:", prompt)
        self.assertIn("actor_lr_multiplier: 0.5 | 0.75", prompt)
        self.assertIn("clip_coef: 0.08 | 0.12", prompt)

    def test_no_action_instruction_is_not_self_contradictory(self) -> None:
        prompt = build_analysis_prompt(manager_context())

        self.assertNotIn("better than the strongest alternative and better than no_action", prompt)
        self.assertIn("If you choose no_action", prompt)

    def test_absent_metrics_are_dropped_rather_than_shown_as_null(self) -> None:
        prompt = build_analysis_prompt(manager_context())

        self.assertNotIn("null", prompt)
        self.assertNotIn("sampled_action_saturation_fraction", prompt)
        self.assertNotIn("collision -", prompt)

    def test_manager_observation_has_deliberate_sections_only(self) -> None:
        observation = build_manager_observation(manager_context())

        self.assertEqual(
            list(observation),
            [
                "run_progress",
                "current_performance",
                "recent_history",
                "ppo_health",
                "best_checkpoint",
                "decision_history",
            ],
        )
        self.assertNotIn("safety_bounds", observation)
        self.assertNotIn("metric_definitions", observation)
        self.assertNotIn("private_marker", json.dumps(observation))

    def test_current_snapshot_is_not_repeated_in_history_or_checkpoint(self) -> None:
        context = manager_context()
        current = dict(context["latest_telemetry"])
        context["recent_telemetry"] = [current]
        context["best_checkpoint"] = {
            "available": True,
            "metrics": {
                **current,
                "eval_safe_success_mean": 0.321,
                "eval_collision_mean": 0.123,
            },
        }

        observation = build_manager_observation(context)

        self.assertEqual(observation["recent_history"], [])
        self.assertEqual(
            observation["best_checkpoint"],
            {"available": True, "update": 20, "is_current_checkpoint": True},
        )

    def test_prior_action_outcomes_are_factual_and_multi_horizon(self) -> None:
        context = manager_context()
        context["latest_telemetry"].update(
            {
                "update": 40,
                "eval_safe_success_mean": 0.5,
                "stochastic_eval_safe_success_mean": 0.45,
                "eval_collision_mean": 0.1,
                "stochastic_eval_collision_mean": 0.15,
            }
        )
        context["recent_telemetry"] = [
            {
                "update": 30,
                "eval_safe_success_mean": 0.4,
                "stochastic_eval_safe_success_mean": 0.35,
                "eval_collision_mean": 0.2,
                "stochastic_eval_collision_mean": 0.25,
            }
        ]
        context["decision_history"] = [
            {
                "decision_update": 20,
                "selected_control": "actor_lr",
                "selected_policy": {"actor_lr_multiplier": 1.25},
                "telemetry_before": {
                    "eval_safe_success_mean": 0.3,
                    "stochastic_eval_safe_success_mean": 0.25,
                    "eval_collision_mean": 0.3,
                    "stochastic_eval_collision_mean": 0.35,
                },
            }
        ]

        history = build_manager_observation(context)["decision_history"]
        outcomes = history[0]["observed_evaluations"]

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(outcomes[0]["updates_after_action"], 10)
        self.assertAlmostEqual(outcomes[0]["deterministic_safe_success_delta"], 0.1)
        self.assertAlmostEqual(outcomes[1]["stochastic_collision_delta"], -0.2)
        self.assertNotIn("assessment", json.dumps(history))

    def test_clamped_multiplier_results_are_not_duplicated(self) -> None:
        recipe = TrainingRecipe(
            actor_lr=1.5e-3,
            critic_lr=1e-3,
            entropy_coef=0.08,
            clip_coef=0.2,
            update_epochs=6,
            target_kl=0.03,
        )

        actions = allowed_actions(recipe, allow_rollback=False, allow_stop=False)

        self.assertEqual(actions["actor_lr"]["actor_lr_multiplier"], [0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0])
        self.assertEqual(actions["entropy"]["entropy_coef_multiplier"], [0.25, 0.5, 0.75, 0.9, 1.1, 1.25])

    def test_two_stage_formatter_receives_only_analysis_actions_and_feedback(self) -> None:
        prompt = build_prompt(
            manager_context(),
            validation_feedback="policy must be an object",
            expert_analysis="Select no_action with an empty policy.",
        )

        self.assertIn("Select no_action with an empty policy.", prompt)
        self.assertIn('"no_action": {}', prompt)
        self.assertIn("policy must be an object", prompt)
        self.assertNotIn("UNSELECTED_TRAINER_FIELD", prompt)
        self.assertNotIn("BOUNDS_MUST_NOT_REACH_FORMATTER", prompt)
        self.assertNotIn("Current recipe:", prompt)
        self.assertNotIn("Context:", prompt)

    def test_compact_payload_passes_runtime_validation(self) -> None:
        decision, error = validate_response(no_action_payload(), manager_context())

        self.assertIsNone(error)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["selected_action"]["policy_update"], {})

    def test_formatter_gets_one_retry_with_only_mechanical_feedback(self) -> None:
        invalid = {"selected_action": no_action_payload()["selected_action"], "watch_next": []}
        valid = no_action_payload()
        manager = RandomizedReacherManager()

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("manager.openai_chat_text", return_value="Select no_action."),
            patch(
                "manager.openai_chat_json",
                side_effect=[json.dumps(invalid), json.dumps(valid)],
            ) as formatter,
        ):
            decision = manager.decide(manager_context())

        self.assertEqual(formatter.call_count, 2)
        retry_prompt = formatter.call_args_list[1].kwargs["prompt"]
        self.assertIn("missing keys: ['reason']", retry_prompt)
        self.assertNotIn("UNSELECTED_TRAINER_FIELD", retry_prompt)
        self.assertEqual(decision["selected_action"]["name"], "no_action")
        self.assertEqual(decision["source"], "openai")
        self.assertIn("## 5. AVAILABLE ACTIONS", decision["stage1_prompt"])
        self.assertEqual(len(decision["stage2_attempts"]), 2)
        self.assertIn("raw_response", decision["stage2_attempts"][0])
        self.assertEqual(decision["stage2_attempts"][0]["validation_error"], "missing keys: ['reason']")
        self.assertIsNone(decision["stage2_attempts"][1]["validation_error"])

    def test_action_surface_contains_only_executable_runtime_actions(self) -> None:
        recipe = TrainingRecipe(
            actor_lr=3e-4,
            critic_lr=1e-3,
            entropy_coef=0.01,
            clip_coef=0.2,
            update_epochs=6,
            target_kl=0.03,
        )
        actions = allowed_actions(recipe, allow_rollback=False, allow_stop=False)

        self.assertNotIn("request_extra_eval", actions)
        self.assertNotIn("rollback_to_best", actions)
        self.assertNotIn("rollback_and_consolidate", actions)
        self.assertNotIn("stop_training", actions)

        enabled = allowed_actions(recipe, allow_rollback=True, allow_stop=True)
        self.assertIn("rollback_to_best", enabled)
        self.assertIn("rollback_and_consolidate", enabled)
        self.assertIn("stop_training", enabled)

    def test_no_offered_value_is_a_no_op_after_clamping(self) -> None:
        """Every value on the action surface must actually change the recipe.

        Multiplier 1.0 and values equal to the current setting must not be offered,
        because they would clamp to no change and waste the intervention.
        """
        grid = {
            "actor_lr": [1e-5, 3e-4, 3e-3],
            "critic_lr": [1e-5, 3e-3],
            "entropy_coef": [0.0, 0.01, 0.1],
            "clip_coef": [0.03, 0.2, 0.3],
            "update_epochs": [1, 4, 8],
            "target_kl": [0.003, 0.1],
            "collision_penalty": [1.0, 50.0],
            "success_bonus": [3.0, 10.0],
            "progress_scale": [1.0, 5.0],
        }
        for combo in itertools.product(*grid.values()):
            recipe = TrainingRecipe(**dict(zip(grid, combo)))
            recipe.clamp()
            surface = allowed_actions(recipe, allow_rollback=True, allow_stop=True)
            for action, params in surface.items():
                if not isinstance(params, dict):
                    continue
                for key, values in params.items():
                    grouped = values.items() if key in {"two_or_three_of", "exactly_one_of"} else [(key, values)]
                    for control, options in grouped:
                        for option in options:
                            updated = apply_recipe_update(recipe, {control: option})
                            self.assertNotEqual(
                                asdict(updated),
                                asdict(recipe),
                                f"{action}.{control}={option} is a no-op for {asdict(recipe)}",
                            )

    def test_checkpoint_selection_uses_both_eval_modes_and_collision(self) -> None:
        safer = {
            "eval_safe_success_mean": 0.7,
            "stochastic_eval_safe_success_mean": 0.65,
            "eval_collision_mean": 0.05,
            "stochastic_eval_collision_mean": 0.1,
            "eval_final_distance_mean": 0.2,
        }
        reckless = {
            "eval_safe_success_mean": 0.8,
            "stochastic_eval_safe_success_mean": 0.75,
            "eval_collision_mean": 0.3,
            "stochastic_eval_collision_mean": 0.35,
            "eval_final_distance_mean": 0.1,
        }
        safer.update(checkpoint_robust_metrics(safer))
        reckless.update(checkpoint_robust_metrics(reckless))

        self.assertGreater(checkpoint_selection_key(safer), checkpoint_selection_key(reckless))

    def test_stop_criterion_requires_every_mode_and_difficulty_to_be_robust(self) -> None:
        row = {
            f"{prefix}_{difficulty}_{metric}": 0.9 if metric == "safe_success" else 0.1
            for prefix in ("eval", "stochastic_eval")
            for difficulty in ("easy", "medium", "hard")
            for metric in ("safe_success", "collision")
        }

        self.assertTrue(meets_robust_stop_criterion(row))
        row["stochastic_eval_hard_safe_success"] = 0.8
        self.assertFalse(meets_robust_stop_criterion(row))

    def test_live_rollback_restores_state_but_retains_active_recipe(self) -> None:
        model = ActorCritic(obs_dim=14, action_dim=2, initial_logstd=-0.7)
        optimizer = torch.optim.Adam(
            [
                {"params": list(model.actor_mean.parameters()) + [model.actor_logstd], "lr": 3e-3},
                {"params": model.critic.parameters(), "lr": 3e-3},
            ]
        )
        checkpoint_recipe = TrainingRecipe(
            actor_lr=3e-3,
            critic_lr=3e-3,
            entropy_coef=0.08,
            clip_coef=0.3,
            update_epochs=8,
            target_kl=0.1,
            collision_penalty=1.0,
            success_bonus=10.0,
            progress_scale=5.0,
        )
        active_recipe = TrainingRecipe(
            actor_lr=7.5e-4,
            critic_lr=1e-3,
            entropy_coef=0.01,
            clip_coef=0.16,
            update_epochs=4,
            target_kl=0.03,
            collision_penalty=10.0,
            success_bonus=8.0,
            progress_scale=3.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "checkpoint.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "recipe": asdict(checkpoint_recipe),
                },
                path,
            )
            restored_recipe, _ = load_checkpoint(
                path,
                model,
                optimizer,
                torch.device("cpu"),
                recipe_override=active_recipe,
            )

        self.assertEqual(asdict(restored_recipe), asdict(active_recipe))
        self.assertEqual(optimizer.param_groups[0]["lr"], active_recipe.actor_lr)
        self.assertEqual(optimizer.param_groups[1]["lr"], active_recipe.critic_lr)


if __name__ == "__main__":
    unittest.main()
