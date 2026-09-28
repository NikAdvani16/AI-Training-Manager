from __future__ import annotations

import json
import math
import os
import textwrap
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_LLM_MODEL = "gpt-5.4"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_FORMATTER_MODEL = "gpt-5.4-mini"
DEFAULT_FORMATTER_TEMPERATURE = 0.1

ACTION_NAMES = {
    "no_action",
    "actor_lr",
    "critic_lr",
    "entropy",
    "clip_range",
    "ppo_epochs",
    "target_kl",
    "collision_penalty",
    "success_bonus",
    "progress_scale",
    "rollback_to_best",
    "rollback_and_consolidate",
    "stop_training",
    "combined_policy",
}

ACTION_DESCRIPTIONS = {
    "no_action": (
        "Continue training unchanged for one interval. This does not preserve or "
        "restore weights; PPO keeps updating the current policy."
    ),
    "actor_lr": "Scale the actor learning rate by the chosen multiplier.",
    "critic_lr": (
        "Scale the critic learning rate by the chosen multiplier. Affects value-function "
        "fit and therefore advantage quality, not the policy directly."
    ),
    "entropy": (
        "Scale the entropy coefficient by the chosen multiplier, changing how strongly "
        "PPO is pushed to keep the action distribution wide."
    ),
    "clip_range": (
        "Set the PPO clip coefficient. Larger values allow bigger policy changes per "
        "update; smaller values constrain them."
    ),
    "ppo_epochs": (
        "Set how many optimization epochs run per batch. More epochs extract more from "
        "each batch but increase the risk of over-fitting to it."
    ),
    "target_kl": (
        "Set the KL threshold that early-stops an update. Only binds when it is close to "
        "the observed approx_kl."
    ),
    "collision_penalty": "Set the reward penalty applied when the arm collides with the obstacle.",
    "success_bonus": "Set the reward bonus granted for reaching the target without colliding.",
    "progress_scale": "Set the weight on the shaping term that rewards closing distance to the target.",
    "rollback_to_best": (
        "Restore policy and optimizer state from the best checkpoint while retaining "
        "the current manager-controlled recipe."
    ),
    "rollback_and_consolidate": (
        "Restore policy and optimizer state from the best checkpoint, retain the current "
        "manager-controlled recipe, and apply exactly one additional conservative control "
        "change."
    ),
    "stop_training": "End training now and keep the current policy as final.",
    "combined_policy": (
        "Apply two or three of the control changes above together as a single decision, "
        "using only values listed for those individual controls."
    ),
}

ACTOR_LR_MULTIPLIERS = [0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
CRITIC_LR_MULTIPLIERS = [0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0]
ENTROPY_MULTIPLIERS = [0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
CLIP_VALUES = [0.03, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3]
PPO_EPOCH_VALUES = [1, 2, 3, 4, 6, 8]
TARGET_KL_VALUES = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
COLLISION_PENALTY_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
SUCCESS_BONUS_VALUES = [2.5, 3.0, 5.0, 8.0, 10.0]
PROGRESS_SCALE_VALUES = [1.0, 2.0, 3.0, 5.0]

BOUNDS = {
    "actor_lr": [1e-5, 3e-3],
    "critic_lr": [1e-5, 3e-3],
    "entropy_coef": [0.0, 0.1],
    "clip_coef": [0.03, 0.3],
    "update_epochs": [1, 8],
    "target_kl": [0.003, 0.1],
    "collision_penalty": [1.0, 50.0],
    "success_bonus": [2.5, 10.0],
    "progress_scale": [1.0, 5.0],
}


@dataclass
class TrainingRecipe:
    actor_lr: float
    critic_lr: float
    entropy_coef: float
    clip_coef: float
    update_epochs: int
    target_kl: float
    collision_penalty: float = 5.0
    success_bonus: float = 5.0
    progress_scale: float = 3.0

    def clamp(self) -> None:
        self.actor_lr = float(min(max(self.actor_lr, BOUNDS["actor_lr"][0]), BOUNDS["actor_lr"][1]))
        self.critic_lr = float(min(max(self.critic_lr, BOUNDS["critic_lr"][0]), BOUNDS["critic_lr"][1]))
        self.entropy_coef = float(min(max(self.entropy_coef, BOUNDS["entropy_coef"][0]), BOUNDS["entropy_coef"][1]))
        self.clip_coef = float(min(max(self.clip_coef, BOUNDS["clip_coef"][0]), BOUNDS["clip_coef"][1]))
        self.update_epochs = int(min(max(int(self.update_epochs), BOUNDS["update_epochs"][0]), BOUNDS["update_epochs"][1]))
        self.target_kl = float(min(max(self.target_kl, BOUNDS["target_kl"][0]), BOUNDS["target_kl"][1]))
        self.collision_penalty = float(
            min(max(self.collision_penalty, BOUNDS["collision_penalty"][0]), BOUNDS["collision_penalty"][1])
        )
        self.success_bonus = float(min(max(self.success_bonus, BOUNDS["success_bonus"][0]), BOUNDS["success_bonus"][1]))
        self.progress_scale = float(min(max(self.progress_scale, BOUNDS["progress_scale"][0]), BOUNDS["progress_scale"][1]))


def openai_chat(
    api_key: str,
    model: str,
    temperature: float,
    prompt: str,
    timeout: int = 120,
    *,
    json_mode: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    def post(request_payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        data = post(payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and '"param": "temperature"' in body and "temperature" in payload:
            retry_payload = dict(payload)
            retry_payload.pop("temperature", None)
            try:
                data = post(retry_payload)
            except urllib.error.HTTPError as retry_exc:
                retry_body = retry_exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenAI HTTP {retry_exc.code}: {retry_body}") from retry_exc
        else:
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {body}") from exc
    return data["choices"][0]["message"]["content"]


def openai_chat_json(api_key: str, model: str, temperature: float, prompt: str, timeout: int = 120) -> str:
    return openai_chat(api_key, model, temperature, prompt, timeout, json_mode=True)


def openai_chat_text(api_key: str, model: str, temperature: float, prompt: str, timeout: int = 120) -> str:
    return openai_chat(api_key, model, temperature, prompt, timeout, json_mode=False)


def _effective_multipliers(current: float, bounds_key: str, multipliers: list[float]) -> list[float]:
    low, high = BOUNDS[bounds_key]
    selected_by_result: dict[float, float] = {}
    for multiplier in multipliers:
        next_value = min(max(float(current) * float(multiplier), low), high)
        if abs(next_value - float(current)) <= 1e-12:
            continue
        result_key = round(next_value, 15)
        candidate = float(multiplier)
        previous = selected_by_result.get(result_key)
        if previous is None or abs(math.log(candidate)) < abs(math.log(previous)):
            selected_by_result[result_key] = candidate
    selected = set(selected_by_result.values())
    return [float(multiplier) for multiplier in multipliers if float(multiplier) in selected]


def _effective_values(current: float | int, values: list[float | int]) -> list[float | int]:
    out = []
    for value in values:
        if abs(float(value) - float(current)) > 1e-12:
            out.append(int(value) if isinstance(value, int) else float(value))
    return out


def _with_nonempty_choices(surface: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for action, spec in surface.items():
        if spec == {}:
            out[action] = spec
            continue
        if "exactly_one_of" in spec:
            choices = {k: v for k, v in spec["exactly_one_of"].items() if v}
            if choices:
                out[action] = {"exactly_one_of": choices}
            continue
        if "two_or_three_of" in spec:
            choices = {k: v for k, v in spec["two_or_three_of"].items() if v}
            if len(choices) >= 2:
                out[action] = {"two_or_three_of": choices}
            continue
        choices = {k: v for k, v in spec.items() if v}
        if choices:
            out[action] = choices
    return out


def allowed_actions(
    recipe: TrainingRecipe | None = None,
    *,
    allow_rollback: bool = True,
    allow_stop: bool = True,
) -> dict[str, Any]:
    if recipe is None:
        actor_lr_multipliers = ACTOR_LR_MULTIPLIERS
        critic_lr_multipliers = CRITIC_LR_MULTIPLIERS
        entropy_multipliers = ENTROPY_MULTIPLIERS
        clip_values = CLIP_VALUES
        ppo_epoch_values = PPO_EPOCH_VALUES
        target_kl_values = TARGET_KL_VALUES
        collision_penalty_values = COLLISION_PENALTY_VALUES
        success_bonus_values = SUCCESS_BONUS_VALUES
        progress_scale_values = PROGRESS_SCALE_VALUES
    else:
        actor_lr_multipliers = _effective_multipliers(recipe.actor_lr, "actor_lr", ACTOR_LR_MULTIPLIERS)
        critic_lr_multipliers = _effective_multipliers(recipe.critic_lr, "critic_lr", CRITIC_LR_MULTIPLIERS)
        entropy_multipliers = _effective_multipliers(recipe.entropy_coef, "entropy_coef", ENTROPY_MULTIPLIERS)
        clip_values = _effective_values(recipe.clip_coef, CLIP_VALUES)
        ppo_epoch_values = _effective_values(recipe.update_epochs, PPO_EPOCH_VALUES)
        target_kl_values = _effective_values(recipe.target_kl, TARGET_KL_VALUES)
        collision_penalty_values = _effective_values(recipe.collision_penalty, COLLISION_PENALTY_VALUES)
        success_bonus_values = _effective_values(recipe.success_bonus, SUCCESS_BONUS_VALUES)
        progress_scale_values = _effective_values(recipe.progress_scale, PROGRESS_SCALE_VALUES)

    surface = {
        "no_action": {},
        "actor_lr": {"actor_lr_multiplier": actor_lr_multipliers},
        "critic_lr": {"critic_lr_multiplier": critic_lr_multipliers},
        "entropy": {"entropy_coef_multiplier": entropy_multipliers},
        "clip_range": {"clip_coef": clip_values},
        "ppo_epochs": {"update_epochs": ppo_epoch_values},
        "target_kl": {"target_kl": target_kl_values},
        "collision_penalty": {"collision_penalty": collision_penalty_values},
        "success_bonus": {"success_bonus": success_bonus_values},
        "progress_scale": {"progress_scale": progress_scale_values},
        "combined_policy": {
            "two_or_three_of": {
                "actor_lr_multiplier": actor_lr_multipliers,
                "critic_lr_multiplier": critic_lr_multipliers,
                "entropy_coef_multiplier": entropy_multipliers,
                "clip_coef": clip_values,
                "update_epochs": ppo_epoch_values,
                "target_kl": target_kl_values,
                "collision_penalty": collision_penalty_values,
                "success_bonus": success_bonus_values,
                "progress_scale": progress_scale_values,
            }
        },
    }
    if allow_rollback:
        surface["rollback_to_best"] = {}
        surface["rollback_and_consolidate"] = {
            "exactly_one_of": {
                "actor_lr_multiplier": [m for m in actor_lr_multipliers if m < 1.0],
                "critic_lr_multiplier": [m for m in critic_lr_multipliers if m < 1.0],
                "entropy_coef_multiplier": [m for m in entropy_multipliers if m < 1.0],
                "clip_coef": [v for v in clip_values if float(v) <= 0.16],
                "update_epochs": [v for v in ppo_epoch_values if int(v) <= 4],
                "target_kl": [v for v in target_kl_values if float(v) <= 0.03],
                "collision_penalty": [v for v in collision_penalty_values if float(v) >= 10.0],
                "success_bonus": [v for v in success_bonus_values if float(v) <= 8.0],
                "progress_scale": [v for v in progress_scale_values if float(v) <= 3.0],
            }
        }
    if allow_stop:
        surface["stop_training"] = {}
    return _with_nonempty_choices(surface)


def _allowed_number(value: Any, allowed: list[float | int]) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return any(abs(float(value) - float(candidate)) <= 1e-12 for candidate in allowed)


def _validators_for_action(surface: dict[str, Any], action: str) -> dict[str, list[float | int]]:
    validators: dict[str, list[float | int]] = {}
    spec = surface.get(action, {})
    if not isinstance(spec, dict):
        return validators
    if "exactly_one_of" in spec:
        choices = spec["exactly_one_of"]
    elif "two_or_three_of" in spec:
        choices = spec["two_or_three_of"]
    else:
        choices = spec
    for key, values in choices.items():
        validators.setdefault(key, [])
        for value in values:
            if not _allowed_number(value, validators[key]):
                validators[key].append(value)
    return validators


def normalize_policy(
    action: str,
    policy_obj: Any,
    action_surface: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if action not in ACTION_NAMES:
        return None, f"selected_action.name must be one of {sorted(ACTION_NAMES)}"
    if action_surface is not None and action not in action_surface:
        return None, f"{action} is not currently available because it has no effective choices"
    if not isinstance(policy_obj, dict):
        return None, "selected_action.policy must be an object"
    if action in {"no_action", "rollback_to_best", "stop_training"}:
        if policy_obj:
            return None, f"{action} policy must be empty"
        return {}, None

    expected = {
        "actor_lr": {"actor_lr_multiplier"},
        "critic_lr": {"critic_lr_multiplier"},
        "entropy": {"entropy_coef_multiplier"},
        "clip_range": {"clip_coef"},
        "ppo_epochs": {"update_epochs"},
        "target_kl": {"target_kl"},
        "collision_penalty": {"collision_penalty"},
        "success_bonus": {"success_bonus"},
        "progress_scale": {"progress_scale"},
    }
    if action in expected and set(policy_obj) != expected[action]:
        return None, f"{action} policy must contain exactly {sorted(expected[action])}"
    if action == "rollback_and_consolidate" and len(policy_obj) != 1:
        return None, "rollback_and_consolidate must include exactly one conservative recipe key"
    if action == "combined_policy" and len(policy_obj) not in {2, 3}:
        return None, "combined_policy must include exactly two or three recipe keys"

    normalized: dict[str, Any] = {}
    validators = (
        _validators_for_action(action_surface, action)
        if action_surface is not None
        else {
            "actor_lr_multiplier": ACTOR_LR_MULTIPLIERS,
            "critic_lr_multiplier": CRITIC_LR_MULTIPLIERS,
            "entropy_coef_multiplier": ENTROPY_MULTIPLIERS,
            "clip_coef": CLIP_VALUES,
            "update_epochs": PPO_EPOCH_VALUES,
            "target_kl": TARGET_KL_VALUES,
            "collision_penalty": COLLISION_PENALTY_VALUES,
            "success_bonus": SUCCESS_BONUS_VALUES,
            "progress_scale": PROGRESS_SCALE_VALUES,
        }
    )
    for key, value in policy_obj.items():
        if key not in validators:
            return None, f"unsupported policy key: {key}"
        if not _allowed_number(value, validators[key]):
            return None, f"{key} must be one of {validators[key]}"
        normalized[key] = int(value) if key == "update_epochs" else float(value)
    return normalized, None


def apply_recipe_update(recipe: TrainingRecipe, update: dict[str, Any]) -> TrainingRecipe:
    out = TrainingRecipe(**asdict(recipe))
    out.actor_lr *= float(update.get("actor_lr_multiplier", 1.0))
    out.critic_lr *= float(update.get("critic_lr_multiplier", 1.0))
    out.entropy_coef *= float(update.get("entropy_coef_multiplier", 1.0))
    if "clip_coef" in update:
        out.clip_coef = float(update["clip_coef"])
    if "update_epochs" in update:
        out.update_epochs = int(update["update_epochs"])
    if "target_kl" in update:
        out.target_kl = float(update["target_kl"])
    if "collision_penalty" in update:
        out.collision_penalty = float(update["collision_penalty"])
    if "success_bonus" in update:
        out.success_bonus = float(update["success_bonus"])
    if "progress_scale" in update:
        out.progress_scale = float(update["progress_scale"])
    out.clamp()
    return out


def validate_response(payload: Any, context: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(payload, dict):
        return None, "response must be a JSON object"
    required = {"selected_action", "reason", "watch_next"}
    missing = sorted(required - set(payload))
    if missing:
        return None, f"missing keys: {missing}"
    selected = payload.get("selected_action")
    if not isinstance(selected, dict):
        return None, "selected_action must be an object"
    selected_required = {"name", "policy", "expected_outcome", "success_criteria"}
    selected_missing = sorted(selected_required - set(selected))
    if selected_missing:
        return None, f"selected_action is missing keys: {selected_missing}"
    for key in ("expected_outcome", "success_criteria"):
        if not isinstance(selected.get(key), str):
            return None, f"selected_action.{key} must be a string"
    if not isinstance(payload.get("reason"), str):
        return None, "reason must be a string"
    watch_next = payload.get("watch_next")
    if not isinstance(watch_next, list) or not all(isinstance(item, str) for item in watch_next):
        return None, "watch_next must be an array of strings"
    action = selected.get("name")
    action_surface = context.get("available_actions")
    policy_update, error = normalize_policy(
        str(action),
        selected.get("policy", {}),
        action_surface if isinstance(action_surface, dict) else None,
    )
    if error:
        return None, error
    if action == "stop_training" and not context["run_state_constraints"]["stop_training_allowed"]:
        return None, "stop_training is not currently allowed"
    if action in {"rollback_to_best", "rollback_and_consolidate"} and not context["run_state_constraints"][
        "rollback_available"
    ]:
        return None, f"{action} requires an available checkpoint"

    current_recipe = TrainingRecipe(**context["current_recipe"])
    next_recipe = apply_recipe_update(current_recipe, policy_update or {})
    if action not in {"no_action", "rollback_to_best", "stop_training"}:
        if asdict(current_recipe) == asdict(next_recipe):
            return None, "selected action has no effect after bounds"

    selected["policy_update"] = policy_update or {}
    payload["selected_action"] = selected
    payload["valid"] = True
    return payload, None


def fallback_decision(reason: str) -> dict[str, Any]:
    return {
        "selected_action": {
            "name": "no_action",
            "policy": {},
            "policy_update": {},
            "expected_outcome": "Training continues unchanged.",
            "success_criteria": "No crash; next telemetry clarifies state.",
        },
        "reason": reason,
        "watch_next": ["eval_safe_success_mean", "eval_collision_mean", "approx_kl", "clip_fraction"],
        "valid": True,
        "source": "fallback",
    }


class RandomizedReacherManager:
    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        formatter_model: str = DEFAULT_FORMATTER_MODEL,
        formatter_temperature: float = DEFAULT_FORMATTER_TEMPERATURE,
    ):
        self.model = model
        self.temperature = float(temperature)
        self.formatter_model = formatter_model
        self.formatter_temperature = float(formatter_temperature)

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return fallback_decision("OPENAI_API_KEY is not set")
        analysis_prompt: str | None = None
        try:
            analysis_prompt = build_analysis_prompt(context)
            analysis_text = openai_chat_text(api_key, self.model, self.temperature, analysis_prompt, timeout=180)
        except Exception as exc:  # noqa: BLE001 - logged as fallback reason.
            fallback = fallback_decision(f"OpenAI reasoning call failed: {exc}")
            if analysis_prompt is not None:
                fallback["stage1_prompt"] = analysis_prompt
            return fallback

        decision: dict[str, Any] | None = None
        validation_feedback: str | None = None
        formatter_attempts: list[dict[str, Any]] = []
        for attempt_index in range(2):
            prompt = build_prompt(context, validation_feedback, expert_analysis=analysis_text)
            attempt_record: dict[str, Any] = {
                "attempt": attempt_index + 1,
                "prompt": prompt,
                "validation_feedback_in": validation_feedback,
            }
            try:
                raw = openai_chat_json(
                    api_key,
                    self.formatter_model,
                    self.formatter_temperature,
                    timeout=120,
                    prompt=prompt,
                )
                attempt_record["raw_response"] = raw
                parsed = json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                validation_feedback = f"response was not valid JSON or the formatter request failed: {exc}"
                attempt_record["validation_error"] = validation_feedback
                formatter_attempts.append(attempt_record)
                continue
            decision, error = validate_response(parsed, context)
            attempt_record["validation_error"] = error
            formatter_attempts.append(attempt_record)
            if error is None:
                break
            validation_feedback = error

        if decision is None:
            fallback = fallback_decision(
                f"Formatter failed mechanical validation after one retry: {validation_feedback or 'unknown error'}"
            )
            fallback["expert_analysis"] = analysis_text
            fallback["stage1_prompt"] = analysis_prompt
            fallback["stage2_attempts"] = formatter_attempts
            return fallback
        decision["source"] = "openai"
        decision["llm_model"] = self.model
        decision["temperature"] = self.temperature
        decision["manager_mode"] = "two_stage"
        decision["reasoning_model"] = self.model
        decision["reasoning_temperature"] = self.temperature
        decision["formatter_model"] = self.formatter_model
        decision["formatter_temperature"] = self.formatter_temperature
        decision["expert_analysis"] = analysis_text
        decision["stage1_prompt"] = analysis_prompt
        decision["stage2_attempts"] = formatter_attempts
        return decision


def _json_clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return value


def _mean_metric(evals: dict[str, Any], key: str) -> float | None:
    values = []
    for metrics in evals.values():
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return float(sum(values) / len(values)) if values else None


def _compact_eval(evals: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    overall = {
        "safe_success_rate": summary.get("safe_success_mean"),
        "collision_episode_rate": summary.get("collision_mean"),
        "mean_final_distance": summary.get("final_distance_mean"),
        "mean_min_distance": summary.get("min_distance_mean"),
        "mean_episode_length": summary.get("episode_length_mean"),
        "mean_steps_to_success": summary.get("steps_to_success_mean"),
        "mean_action_norm": _mean_metric(evals, "mean_action_norm"),
        "sampled_action_saturation_fraction": _mean_metric(evals, "sampled_action_saturation_fraction"),
    }
    compact: dict[str, Any] = {"overall": overall}
    for difficulty in ("easy", "medium", "hard"):
        metrics = evals.get(difficulty, {})
        if not isinstance(metrics, dict) or not metrics:
            continue
        compact[difficulty] = {
            "safe_success_rate": metrics.get("safe_success_rate"),
            "collision_episode_rate": metrics.get("collision_episode_rate"),
            "mean_final_distance": metrics.get("mean_final_distance"),
            "mean_min_distance": metrics.get("mean_min_distance"),
            "mean_steps_to_success": metrics.get("mean_steps_to_success"),
        }
    return _json_clean(compact)


_RECIPE_TELEMETRY_KEYS = (
    "actor_lr",
    "critic_lr",
    "entropy_coef",
    "clip_coef",
    "update_epochs",
    "target_kl",
    "collision_penalty",
    "success_bonus",
    "progress_scale",
)


def _recipe_from_telemetry(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _RECIPE_TELEMETRY_KEYS if key in row}


def _compact_history_row(row: dict[str, Any]) -> dict[str, Any]:
    return _json_clean(
        {
            "update": row.get("update"),
            "global_step": row.get("global_step"),
            "train_safe_success_rate": row.get("train_safe_success_rate"),
            "train_collision_episode_rate": row.get("train_collision_episode_rate"),
            "deterministic_safe_success_rate": row.get("eval_safe_success_mean"),
            "deterministic_collision_episode_rate": row.get("eval_collision_mean"),
            "deterministic_mean_final_distance": row.get("eval_final_distance_mean"),
            "deterministic_mean_min_distance": row.get("eval_min_distance_mean"),
            "stochastic_safe_success_rate": row.get("stochastic_eval_safe_success_mean"),
            "stochastic_collision_episode_rate": row.get("stochastic_eval_collision_mean"),
            "checkpoint_robust_score": row.get("checkpoint_robust_score"),
            "ppo_health": {
                "approx_kl": row.get("approx_kl"),
                "kl_to_target": row.get("kl_to_target"),
                "clip_fraction": row.get("clip_fraction"),
                "epochs_ran": row.get("epochs_ran"),
                "explained_variance": row.get("explained_variance"),
                "actor_action_std": row.get("actor_action_std"),
                "sampled_action_saturation_fraction": row.get("sampled_action_saturation_fraction"),
                "policy_update_rel_l2": row.get("policy_update_rel_l2"),
            },
        }
    )


def _compact_best_checkpoint(context: dict[str, Any], latest: dict[str, Any]) -> dict[str, Any]:
    checkpoint = context.get("best_checkpoint", {})
    metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(metrics, dict) or not metrics:
        return {"available": False}
    checkpoint_update = metrics.get("update")
    if checkpoint_update == latest.get("update"):
        current_checkpoint = {
            "available": True,
            "update": checkpoint_update,
            "is_current_checkpoint": True,
        }
        if metrics.get("checkpoint_selection_criterion") is not None:
            current_checkpoint["selection_criterion"] = metrics["checkpoint_selection_criterion"]
        if metrics.get("checkpoint_robust_score") is not None:
            current_checkpoint["robust_score"] = metrics["checkpoint_robust_score"]
        return current_checkpoint
    return _json_clean(
        {
            "available": True,
            "update": checkpoint_update,
            "is_current_checkpoint": False,
            "selection_criterion": metrics.get("checkpoint_selection_criterion"),
            "robust_score": metrics.get("checkpoint_robust_score"),
            "worst_eval_cell_safe_success_rate": metrics.get("checkpoint_worst_eval_cell_safe_success"),
            "worst_eval_cell_collision_rate": metrics.get("checkpoint_worst_eval_cell_collision"),
            "deterministic_safe_success_rate": metrics.get("eval_safe_success_mean"),
            "deterministic_collision_episode_rate": metrics.get("eval_collision_mean"),
            "deterministic_mean_final_distance": metrics.get("eval_final_distance_mean"),
            "deterministic_mean_min_distance": metrics.get("eval_min_distance_mean"),
            "stochastic_safe_success_rate": metrics.get("stochastic_eval_safe_success_mean"),
            "stochastic_collision_episode_rate": metrics.get("stochastic_eval_collision_mean"),
            "actor_action_std": metrics.get("actor_action_std"),
            "recipe_at_checkpoint": _recipe_from_telemetry(metrics),
        }
    )


def _compact_decision_history(context: dict[str, Any]) -> list[dict[str, Any]]:
    telemetry = [row for row in context.get("recent_telemetry", []) if isinstance(row, dict)]
    latest = context.get("latest_telemetry", {})
    if isinstance(latest, dict):
        telemetry.append(latest)
    telemetry = sorted(
        {int(row["update"]): row for row in telemetry if isinstance(row.get("update"), (int, float))}.values(),
        key=lambda row: int(row["update"]),
    )
    decisions = [decision for decision in context.get("decision_history", []) if isinstance(decision, dict)][-4:]
    compact = []
    for decision in decisions:
        update = decision.get("application_update", decision.get("decision_update"))
        before = decision.get("telemetry_before", {})
        observed: list[dict[str, Any]] = []
        if isinstance(before, dict) and isinstance(update, (int, float)):
            before_det_safe = before.get("eval_safe_success_mean")
            before_stoch_safe = before.get("stochastic_eval_safe_success_mean")
            before_det_collision = before.get("eval_collision_mean")
            before_stoch_collision = before.get("stochastic_eval_collision_mean")
            next_recipe_change = next(
                (
                    float(other_update)
                    for other in decisions
                    if other is not decision
                    and (other_update := other.get("application_update", other.get("decision_update")))
                    is not None
                    and isinstance(other_update, (int, float))
                    and float(other_update) > float(update)
                    and other.get("selected_control", other.get("action")) != "no_action"
                ),
                None,
            )
            after_rows = [
                row
                for row in telemetry
                if float(row["update"]) > float(update)
                and (next_recipe_change is None or float(row["update"]) < next_recipe_change)
            ][:3]
            for after in after_rows:
                evaluation_update = after.get("update")

                def delta(after_key: str, before_value: Any) -> float | None:
                    after_value = after.get(after_key)
                    if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
                        return float(after_value) - float(before_value)
                    return None

                observed.append(
                    {
                        "evaluation_update": evaluation_update,
                        "updates_after_action": int(evaluation_update) - int(update),
                        "deterministic_safe_success_delta": delta("eval_safe_success_mean", before_det_safe),
                        "stochastic_safe_success_delta": delta(
                            "stochastic_eval_safe_success_mean", before_stoch_safe
                        ),
                        "deterministic_collision_delta": delta("eval_collision_mean", before_det_collision),
                        "stochastic_collision_delta": delta(
                            "stochastic_eval_collision_mean", before_stoch_collision
                        ),
                    }
                )
        selected_policy = decision.get("selected_policy", {})
        recipe_after = decision.get("recipe_after", {})
        result_key_by_policy_key = {
            "actor_lr_multiplier": "actor_lr",
            "critic_lr_multiplier": "critic_lr",
            "entropy_coef_multiplier": "entropy_coef",
            "clip_coef": "clip_coef",
            "update_epochs": "update_epochs",
            "target_kl": "target_kl",
            "collision_penalty": "collision_penalty",
            "success_bonus": "success_bonus",
            "progress_scale": "progress_scale",
        }
        resulting_values = {
            recipe_key: recipe_after.get(recipe_key)
            for policy_key, recipe_key in result_key_by_policy_key.items()
            if isinstance(selected_policy, dict) and policy_key in selected_policy and recipe_key in recipe_after
        }
        compact.append(
            _json_clean(
                {
                    "request_context_update": decision.get("request_context_update"),
                    "application_update": update,
                    "application_lag_updates": decision.get("application_lag_updates"),
                    "action": decision.get("selected_control", decision.get("action")),
                    "policy": selected_policy,
                    "resulting_values": resulting_values,
                    "reason": decision.get("rationale_then"),
                    "expected_outcome": decision.get("expected_outcome"),
                    "observed_evaluations": observed,
                }
            )
        )
    return compact


# ---------------------------------------------------------------------------
# Rendering helpers. The manager reads aligned tables instead of nested JSON:
# trends are meant to be legible by scanning a column.
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _fmt_rate(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}" if _is_number(value) else "-"


def _fmt_g(value: Any, sig: int = 3) -> str:
    return f"{float(value):.{sig}g}" if _is_number(value) else "-"


def _fmt_delta(value: Any, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}" if _is_number(value) else "-"


def _fmt_int(value: Any) -> str:
    return str(int(value)) if _is_number(value) else "-"


def _shorten(text: Any, limit: int = 200) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "..."


def _has_value(node: Any, key: str) -> bool:
    """True when `key` appears anywhere in `node` with a non-null value."""
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key and value is not None:
                return True
            if _has_value(value, key):
                return True
    elif isinstance(node, list):
        return any(_has_value(item, key) for item in node)
    return False


def _render_table(
    headers: list[str],
    rows: list[list[str]],
    indent: str = "",
    label_columns: int = 1,
) -> str:
    """Render an aligned table, dropping any column that carries no data.

    The leading `label_columns` identify rows rather than carry measurements, so
    the table is suppressed entirely when no data column survives.
    """
    text_rows = [[str(cell) for cell in row] for row in rows]
    if not text_rows:
        return ""
    keep = [index for index in range(len(headers)) if any(row[index] not in ("", "-") for row in text_rows)]
    if not any(index >= label_columns for index in keep):
        return ""
    headers = [headers[index] for index in keep]
    text_rows = [[row[index] for index in keep] for row in text_rows]
    widths = [max([len(headers[i])] + [len(row[i]) for row in text_rows]) for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        parts = [cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(cells)]
        return (indent + "  ".join(parts)).rstrip()

    return "\n".join([line(headers)] + [line(row) for row in text_rows])


def _render_recipe(recipe: dict[str, Any]) -> str:
    order = [
        "actor_lr",
        "critic_lr",
        "entropy_coef",
        "clip_coef",
        "update_epochs",
        "target_kl",
        "collision_penalty",
        "success_bonus",
        "progress_scale",
    ]
    rows = [
        [name, _fmt_int(recipe[name]) if name == "update_epochs" else _fmt_g(recipe[name], 4)]
        for name in order
        if name in recipe
    ]
    return _render_table(["control", "value"], rows)


def _render_eval_matrix(performance: dict[str, Any]) -> str:
    headers = ["mode", "split", "safe_success", "collision", "min_dist", "final_dist", "steps_to_success"]
    rows: list[list[str]] = []
    for mode_key, label in (("deterministic_eval", "deterministic"), ("stochastic_eval", "stochastic")):
        block = performance.get(mode_key) or {}
        for split in ("overall", "easy", "medium", "hard"):
            metrics = block.get(split)
            if not isinstance(metrics, dict) or not metrics:
                continue
            rows.append(
                [
                    label,
                    split,
                    _fmt_rate(metrics.get("safe_success_rate")),
                    _fmt_rate(metrics.get("collision_episode_rate")),
                    _fmt_rate(metrics.get("mean_min_distance")),
                    _fmt_rate(metrics.get("mean_final_distance")),
                    _fmt_rate(metrics.get("mean_steps_to_success"), 1),
                ]
            )
    return _render_table(headers, rows, label_columns=2)


def _render_behavior(performance: dict[str, Any]) -> str:
    headers = ["mode", "mean_action_norm", "mean_episode_length", "action_saturation"]
    rows: list[list[str]] = []
    for mode_key, label in (("deterministic_eval", "deterministic"), ("stochastic_eval", "stochastic")):
        overall = (performance.get(mode_key) or {}).get("overall") or {}
        rows.append(
            [
                label,
                _fmt_rate(overall.get("mean_action_norm")),
                _fmt_rate(overall.get("mean_episode_length"), 1),
                _fmt_rate(overall.get("sampled_action_saturation_fraction")),
            ]
        )
    return _render_table(headers, rows)


def _render_ppo_health(health: dict[str, Any], recipe: dict[str, Any]) -> str:
    epochs_ran = health.get("epochs_ran")
    configured = recipe.get("update_epochs")
    if _is_number(epochs_ran) and _is_number(configured):
        epochs_text = f"{_fmt_int(epochs_ran)} of {_fmt_int(configured)} configured"
    else:
        epochs_text = _fmt_int(epochs_ran)
    rows = [
        ["approx_kl", _fmt_g(health.get("approx_kl"))],
        ["approx_kl / target_kl", _fmt_g(health.get("kl_to_target"))],
        ["clip_fraction", _fmt_g(health.get("clip_fraction"))],
        ["epochs_ran", epochs_text],
        ["explained_variance", _fmt_rate(health.get("explained_variance"))],
        ["actor_action_std", _fmt_g(health.get("actor_action_std"), 4)],
        ["sampled_action_saturation_fraction", _fmt_rate(health.get("sampled_action_saturation_fraction"))],
        ["policy_update_rel_l2", _fmt_g(health.get("policy_update_rel_l2"))],
    ]
    rows = [row for row in rows if row[1] != "-"]
    return _render_table(["metric", "value"], rows)


def _render_history_table(history: list[dict[str, Any]]) -> str:
    headers = [
        "update",
        "det_safe",
        "stoch_safe",
        "det_coll",
        "stoch_coll",
        "min_dist",
        "train_safe",
        "train_coll",
        "approx_kl",
        "clip_frac",
        "epochs",
        "act_std",
        "expl_var",
        "saturation",
    ]
    rows = []
    for row in history:
        health = row.get("ppo_health") or {}
        rows.append(
            [
                _fmt_int(row.get("update")),
                _fmt_rate(row.get("deterministic_safe_success_rate")),
                _fmt_rate(row.get("stochastic_safe_success_rate")),
                _fmt_rate(row.get("deterministic_collision_episode_rate")),
                _fmt_rate(row.get("stochastic_collision_episode_rate")),
                _fmt_rate(row.get("deterministic_mean_min_distance")),
                _fmt_rate(row.get("train_safe_success_rate")),
                _fmt_rate(row.get("train_collision_episode_rate")),
                _fmt_g(health.get("approx_kl")),
                _fmt_g(health.get("clip_fraction")),
                _fmt_int(health.get("epochs_ran")),
                _fmt_g(health.get("actor_action_std"), 4),
                _fmt_rate(health.get("explained_variance")),
                _fmt_rate(health.get("sampled_action_saturation_fraction")),
            ]
        )
    return _render_table(headers, rows)


def _render_decision_history(decisions: list[dict[str, Any]]) -> str:
    if not decisions:
        return "No manager interventions have been applied yet in this run."
    blocks = []
    for decision in decisions:
        update = decision.get("application_update")
        action = decision.get("action", "unknown")
        policy = decision.get("policy") or {}
        resulting = decision.get("resulting_values") or {}
        header = f"update {_fmt_int(update)}: {action}"
        if policy:
            chosen = ", ".join(f"{key}={_fmt_g(value, 4)}" for key, value in policy.items())
            header += f" ({chosen})"
        if resulting:
            became = ", ".join(f"{key}->{_fmt_g(value, 4)}" for key, value in resulting.items())
            header += f" [recipe now: {became}]"
        lines = [header]
        request_update = decision.get("request_context_update")
        lag = decision.get("application_lag_updates")
        if _is_number(request_update) and _is_number(lag) and int(lag) > 0:
            lines.append(
                f"  based on telemetry at update {_fmt_int(request_update)}; "
                f"applied {_fmt_int(lag)} updates later"
            )
        reason = decision.get("reason")
        if reason:
            lines.append(f"  rationale: {_shorten(reason, 400)}")
        expected = decision.get("expected_outcome")
        if expected:
            lines.append(f"  expected: {_shorten(expected, 300)}")
        observed = decision.get("observed_evaluations") or []
        if observed:
            rows = [
                [
                    _fmt_int(entry.get("evaluation_update")),
                    f"+{_fmt_int(entry.get('updates_after_action'))}",
                    _fmt_delta(entry.get("deterministic_safe_success_delta")),
                    _fmt_delta(entry.get("stochastic_safe_success_delta")),
                    _fmt_delta(entry.get("deterministic_collision_delta")),
                    _fmt_delta(entry.get("stochastic_collision_delta")),
                ]
                for entry in observed
            ]
            table = _render_table(
                ["eval_at", "after", "d_det_safe", "d_stoch_safe", "d_det_coll", "d_stoch_coll"],
                rows,
                indent="    ",
            )
            lines.append("  measured change from the telemetry just before this action:")
            lines.append(table)
        else:
            lines.append("  measured: no evaluation has completed since this action.")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_best_checkpoint(best: dict[str, Any], current_recipe: dict[str, Any]) -> str:
    if not best.get("available"):
        return "No best checkpoint has been recorded yet."
    if best.get("is_current_checkpoint"):
        lines = [
            f"The current policy (update {_fmt_int(best.get('update'))}) is itself the best checkpoint so far."
        ]
        if _is_number(best.get("robust_score")):
            lines.append(f"  robust checkpoint score: {_fmt_rate(best.get('robust_score'))}")
        if best.get("selection_criterion"):
            lines.append("  checkpoint criterion: " + str(best["selection_criterion"]))
        return "\n".join(lines)
    lines = [f"Best checkpoint so far: update {_fmt_int(best.get('update'))}"]
    if best.get("selection_criterion"):
        lines.append("  checkpoint criterion: " + str(best["selection_criterion"]))
    robust_rows = [
        [
            _fmt_rate(best.get("robust_score")),
            _fmt_rate(best.get("worst_eval_cell_safe_success_rate")),
            _fmt_rate(best.get("worst_eval_cell_collision_rate")),
        ]
    ]
    robust_table = _render_table(
        ["robust_score", "worst_cell_safe", "worst_cell_collision"],
        robust_rows,
        label_columns=0,
    )
    if robust_table:
        lines.append(robust_table)
    rows = [
        [
            "best checkpoint",
            _fmt_rate(best.get("deterministic_safe_success_rate")),
            _fmt_rate(best.get("stochastic_safe_success_rate")),
            _fmt_rate(best.get("deterministic_collision_episode_rate")),
            _fmt_rate(best.get("stochastic_collision_episode_rate")),
            _fmt_rate(best.get("deterministic_mean_min_distance")),
        ]
    ]
    table = _render_table(
        ["policy", "det_safe", "stoch_safe", "det_coll", "stoch_coll", "min_dist"],
        rows,
    )
    if table:
        lines.append(table)
    checkpoint_recipe = best.get("recipe_at_checkpoint") or {}
    if checkpoint_recipe:
        differences = [
            f"{key}: {_fmt_g(checkpoint_recipe[key], 4)} at checkpoint vs {_fmt_g(current_recipe[key], 4)} now"
            for key in checkpoint_recipe
            if key in current_recipe and checkpoint_recipe[key] != current_recipe[key]
        ]
        if differences:
            lines.append("  recipe differences: " + "; ".join(differences))
        else:
            lines.append("  the recipe is unchanged since that checkpoint.")
    return "\n".join(lines)


def _render_actions(actions: dict[str, Any]) -> str:
    lines = []
    for name in sorted(actions):
        parameters = actions[name] or {}
        if name == "combined_policy":
            controls = parameters.get("two_or_three_of")
            if isinstance(controls, dict):
                lines.append(f"{name}  (choose two or three of: {', '.join(sorted(controls))})")
            else:
                lines.append(name)
        elif isinstance(parameters, dict) and isinstance(parameters.get("exactly_one_of"), dict):
            choices = parameters["exactly_one_of"]
            rendered = "; ".join(
                f"{key}: " + " | ".join(_fmt_g(value, 4) for value in values)
                for key, values in choices.items()
                if isinstance(values, list)
            )
            lines.append(f"{name}  choose exactly one of: {rendered}" if rendered else name)
        elif isinstance(parameters, dict) and parameters:
            rendered = "; ".join(
                f"{key}: " + " | ".join(_fmt_g(value, 4) for value in values)
                for key, values in parameters.items()
                if isinstance(values, list)
            )
            lines.append(f"{name}  {rendered}" if rendered else name)
        else:
            lines.append(name)
        description = ACTION_DESCRIPTIONS.get(name)
        if description:
            lines.append(textwrap.fill(description, width=80, initial_indent="    ", subsequent_indent="    "))
    return "\n".join(lines)


def build_manager_observation(context: dict[str, Any]) -> dict[str, Any]:
    latest = context.get("latest_telemetry", {})
    if not isinstance(latest, dict):
        latest = {}
    deterministic_evals = latest.get("deterministic_eval", {})
    stochastic_evals = latest.get("stochastic_eval", {})
    deterministic_summary = latest.get("deterministic_eval_summary", {})
    stochastic_summary = latest.get("stochastic_eval_summary", {})

    history_rows = [
        row
        for row in context.get("recent_telemetry", [])
        if isinstance(row, dict) and row.get("update") != latest.get("update")
    ]

    compact_history = [_compact_history_row(row) for row in history_rows[-8:]]

    run_plan = context.get("run_plan") or {}
    observation = {
        "run_progress": {
            "update": latest.get("update"),
            "total_updates": run_plan.get("total_updates"),
            "global_step": latest.get("global_step"),
            "total_timesteps": run_plan.get("total_timesteps"),
            "eval_episodes_per_split": run_plan.get("eval_episodes_per_split"),
            "manager_application_mode": run_plan.get("manager_application_mode"),
            "async_lead_updates": run_plan.get("async_lead_updates"),
            "async_max_wait_updates": run_plan.get("async_max_wait_updates"),
            "async_wait_timeout_seconds": run_plan.get("async_wait_timeout_seconds"),
        },
        "current_performance": {
            "train": {
                "safe_success_rate": latest.get("train_safe_success_rate"),
                "collision_episode_rate": latest.get("train_collision_episode_rate"),
                "window_completed_episodes": latest.get("train_window_completed_episodes"),
            },
            "deterministic_eval": _compact_eval(deterministic_evals, deterministic_summary),
            "stochastic_eval": _compact_eval(stochastic_evals, stochastic_summary),
        },
        "recent_history": compact_history,
        "ppo_health": {
            "approx_kl": latest.get("approx_kl"),
            "kl_to_target": latest.get("kl_to_target"),
            "clip_fraction": latest.get("clip_fraction"),
            "epochs_ran": latest.get("epochs_ran"),
            "explained_variance": latest.get("explained_variance"),
            "actor_action_std": latest.get("actor_action_std"),
            "sampled_action_saturation_fraction": latest.get("sampled_action_saturation_fraction"),
            "policy_update_rel_l2": latest.get("policy_update_rel_l2"),
        },
        "best_checkpoint": _compact_best_checkpoint(context, latest),
        "decision_history": _compact_decision_history(context),
    }
    return _json_clean(observation)


def build_analysis_prompt(context: dict[str, Any]) -> str:
    current_recipe = context.get("current_recipe", {}) or {}
    actions = context.get("available_actions", {}) or {}
    observation = build_manager_observation(context)
    performance = observation["current_performance"]
    progress = observation["run_progress"]
    train = performance.get("train") or {}

    reading_notes = [
        "Prefer trends across at least two evaluations. Evaluation noise is real, so do "
        "not infer a trend from one good or bad result, including a new best checkpoint. "
        "A single snapshot can still justify action when several independent metrics and "
        "a clear mechanism agree; otherwise gather evidence by continuing unchanged. Do "
        "not reverse or repeat a change solely because of one noisy evaluation.",
        "Deterministic evaluation runs the mean action and measures the policy's "
        "central behavior. Stochastic evaluation samples from the learned Gaussian "
        "and measures the behavior PPO is actually reinforcing. Both use the same "
        "fixed easy/medium/hard diagnostic scenes at every evaluation, so a gap is not "
        "caused by different scene sets. Stochastic estimates still contain action-"
        "sampling noise.",
        "Training rollout rates come from a rolling window of at most 200 completed "
        "episodes, so they lag the evaluation numbers and move more smoothly.",
        "Optimizer controls (learning rates, clip range, epochs, target KL) act on "
        "the next interval. Reward-weight controls change the learning signal itself "
        "and usually need two or three intervals before their effect is readable.",
        "entropy_coef sets how strongly PPO is pushed to keep exploring. It does not "
        "directly set actor_action_std, which is a learned parameter you cannot edit.",
        "target_kl only constrains an update when it is close to the observed "
        "approx_kl. epochs_ran below the configured value means KL early-stopping is "
        "binding and is already limiting each update.",
        "A best checkpoint shows that those weights evaluated well. It does not show "
        "that the recipe which produced them should continue unchanged.",
        "Prior interventions below list what you expected and what was then measured. "
        "Interpret those measurements together with current telemetry and application "
        "lag. Do not repeat a recent "
        "intervention whose measured outcome did not support its hypothesis unless "
        "something in the current telemetry gives a concrete reason to expect a "
        "different result this time.",
    ]
    if _has_value(observation, "sampled_action_saturation_fraction"):
        reading_notes.insert(
            5,
            "sampled_action_saturation_fraction is the fraction of sampled action "
            "components falling outside the [-1, 1] action bounds before clipping. "
            "When it is high, additional Gaussian noise stops producing "
            "proportionally more varied applied actions.",
        )
    guidance = "\n\n".join(
        textwrap.fill(note, width=80, initial_indent="- ", subsequent_indent="  ")
        for note in reading_notes
    )

    lead = _fmt_int(progress.get("async_lead_updates"))
    max_wait = _fmt_int(progress.get("async_max_wait_updates"))
    timeout = _fmt_g(progress.get("async_wait_timeout_seconds"), 4)
    timing = (
        "Training continues while you reason. Your recommendation is intended to be "
        f"applied about {lead} PPO updates after this telemetry snapshot. If no response "
        f"has arrived by {max_wait} updates after the snapshot, training pauses and waits "
        f"for at most {timeout} seconds before abandoning the call. Judge the supplied "
        "snapshot, but choose an intervention whose rationale remains sensible over that "
        "possible delay."
    )

    sections = [
        "You are an expert reinforcement-learning engineer managing a PPO training run\n"
        "that is currently in progress. You are called at fixed evaluation intervals and\n"
        f"decide what to change, if anything. {timing}",
        "## 1. TASK\n\n"
        "A policy controls a two-link robot arm moving in a two-dimensional plane. At\n"
        "each step it emits two continuous values, one angular velocity per joint, and\n"
        "each is clipped to the range [-1, 1] before the simulator applies it. Every\n"
        "episode randomizes the start pose, the target position, and the obstacle\n"
        "position and radius. An episode is a safe success only if the arm reaches the\n"
        "target without ever colliding; a collision ends the episode with a penalty.\n\n"
        "The goal is a policy that reaches targets safely across the whole task\n"
        "distribution: high safe-success and low collision rates under both\n"
        "deterministic and stochastic evaluation, across easy, medium, and hard scenes.",
        f"## 2. HOW TO READ THIS RUN\n\n{guidance}",
    ]

    update_text = _fmt_int(progress.get("update"))
    if _is_number(progress.get("total_updates")):
        update_text += f" of {_fmt_int(progress.get('total_updates'))}"
    step_text = _fmt_int(progress.get("global_step"))
    if _is_number(progress.get("total_timesteps")):
        step_text += f" of {_fmt_int(progress.get('total_timesteps'))}"
    state_parts = [
        f"## 3. CURRENT STATE  (update {update_text}, global step {step_text})",
        "Current recipe:\n" + _render_recipe(current_recipe),
    ]
    matrix = _render_eval_matrix(performance)
    if matrix:
        caption = "Evaluation on the fixed diagnostic scenes"
        if _is_number(progress.get("eval_episodes_per_split")):
            caption += (
                f" ({_fmt_int(progress.get('eval_episodes_per_split'))} episodes per split,"
                " in each of the two modes)"
            )
        state_parts.append(caption + ":\n" + matrix)
    behavior = _render_behavior(performance)
    if behavior:
        state_parts.append("Evaluation behavior:\n" + behavior)
    train_stats = []
    if _is_number(train.get("safe_success_rate")):
        train_stats.append(f"safe_success {_fmt_rate(train.get('safe_success_rate'))}")
    if _is_number(train.get("collision_episode_rate")):
        train_stats.append(f"collision {_fmt_rate(train.get('collision_episode_rate'))}")
    if _is_number(train.get("window_completed_episodes")):
        train_stats.append(f"window size {_fmt_int(train.get('window_completed_episodes'))} episodes")
    if train_stats:
        state_parts.append(
            "Training rollouts (rolling window of at most 200 completed episodes): "
            + ", ".join(train_stats)
        )
    health = _render_ppo_health(observation["ppo_health"], current_recipe)
    if health:
        state_parts.append("PPO update health on the most recent update:\n" + health)
    sections.append("\n\n".join(state_parts))

    history_parts = ["## 4. HISTORY"]
    history_table = _render_history_table(observation["recent_history"])
    if history_table:
        history_parts.append("Recent evaluations, oldest first:\n" + history_table)
    history_parts.append(_render_best_checkpoint(observation["best_checkpoint"], current_recipe))
    history_parts.append(
        "Your previous interventions and what was measured after them:\n\n"
        + _render_decision_history(observation["decision_history"])
    )
    sections.append("\n\n".join(history_parts))

    sections.append(
        "## 5. AVAILABLE ACTIONS\n\n"
        "Every action listed here is authorized for the current telemetry snapshot. Use\n"
        "only these names and exact values. For asynchronous calls, rollback and stopping\n"
        "eligibility are checked again against fresh telemetry when the response is applied.\n\n"
        + _render_actions(actions)
    )

    sections.append(
        "## 6. YOUR DECISION\n\n"
        "Work through the following in plain English:\n\n"
        "1. What the evidence says about the run's current state and the dominant\n"
        "   problem holding it back.\n\n"
        "2. What your previous interventions predicted, what was actually measured, and\n"
        "   what that rules in or out for this call.\n\n"
        "3. Which actions could address that dominant problem, and why the one you pick\n"
        "   is better than the strongest alternative. If you choose an intervention,\n"
        "   explain why it is better than no_action. If you choose no_action, explain why\n"
        "   the evidence does not yet support an intervention.\n\n"
        "4. What you expect to observe if you are right, and what would show you were\n"
        "   wrong.\n\n"
        "Output exactly one decision. Do not offer alternatives, a ranked list, or a\n"
        "conditional plan such as \"do X, and if that fails do Y\". The next stage only\n"
        "transcribes your decision into JSON; it cannot choose between options, and\n"
        "anything ambiguous is discarded and replaced with no_action.\n\n"
        "Choose a single-control action when one control adequately addresses the causal\n"
        "problem. When two or three controls address that same problem and need to move\n"
        "together, choose combined_policy and give that single bundle. combined_policy is\n"
        "still one decision: bundle only jointly necessary changes, not everything worth\n"
        "trying.\n\n"
        "Copy the action name and every value exactly as written in AVAILABLE ACTIONS.\n"
        "Do not invent values, ranges, or controls.\n\n"
        "End your response with exactly these fields. Each must stand on its own and be\n"
        "understandable without the analysis above:\n\n"
        "Selected action name:\n"
        "Exact policy object:\n"
        "Reason:\n"
        "Expected outcome and evaluation horizon:\n"
        "Success criteria:\n"
        "Metrics to watch next:"
    )
    return "\n\n".join(sections) + "\n"


def build_prompt(
    context: dict[str, Any],
    validation_feedback: str | None = None,
    *,
    expert_analysis: str,
) -> str:
    formatter_schema = {
        "selected_action": {
            "name": "one action name from available_actions",
            "policy": {},
            "expected_outcome": "copied from the expert decision",
            "success_criteria": "copied from the expert decision",
        },
        "reason": "copied from the expert decision",
        "watch_next": ["metrics explicitly mentioned by the expert decision"],
    }
    return f"""
You are a JSON formatter. An expert RL manager has already made the substantive decision. Your job is only to convert that decision into valid JSON.

Do not re-diagnose the run.
Do not choose a different substantive action.
Do not add reasoning that is not present in the expert decision.
Use only action names and policy values present in available_actions.
If the expert decision does not unambiguously specify a valid action and exact policy values, return no_action with an empty policy object. State in reason that the expert decision was ambiguous. Do not attempt to resolve the ambiguity yourself.

If mechanical validation feedback is provided below, correct only that formatting or schema error while preserving the expert decision. If the expert decision cannot be represented validly after applying the feedback, return no_action as described above.

Expert decision:
{expert_analysis}

Available actions:
{json.dumps(context.get("available_actions", {}), sort_keys=True)}

Mechanical validation feedback from the previous formatting attempt, if any:
{validation_feedback or "none"}

Return only valid JSON matching this exact schema:
{json.dumps(formatter_schema, indent=2)}
"""
