from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"


def nonnegative_seed(value: str) -> int:
    seed = int(value)
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer")
    return seed


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_flag(command: list[str], name: str, value: Any) -> None:
    command.extend([f"--{name}", str(value)])


def build_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    common = load_json(CONFIG_DIR / "common.json")
    condition = load_json(CONFIG_DIR / f"{args.condition}.json")
    protocol = common["protocol"]
    ppo = common["ppo"]
    manager = common["manager"]
    recipe = condition["recipe"]
    mode = "manager" if args.manager else "fixed"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / f"{args.condition}_{mode}_seed{args.seed}"

    command = [sys.executable, str(ROOT / "train_ppo.py")]
    for key in (
        "total_updates",
        "num_envs",
        "steps_per_update",
        "minibatch_size",
        "max_episode_steps",
        "success_radius",
        "eval_episodes",
    ):
        add_flag(command, key, protocol[key])
    for key in ("gamma", "gae_lambda", "vf_coef", "max_grad_norm"):
        add_flag(command, key, ppo[key])
    if ppo["anneal_lr"]:
        command.append("--anneal_lr")
    add_flag(command, "eval_interval", protocol["manager_interval"] if args.manager else protocol["fixed_eval_interval"])
    add_flag(command, "final_eval_episodes", protocol["final_eval_episodes_per_difficulty"])
    add_flag(command, "final_eval_seed_base", protocol["final_eval_seed_base"])
    for key, value in recipe.items():
        add_flag(command, key, value)
    add_flag(command, "seed", args.seed)
    add_flag(command, "device", args.device)
    add_flag(command, "out_dir", out_dir)

    if args.manager:
        if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
            raise SystemExit("OPENAI_API_KEY must be set for a manager run")
        add_flag(command, "mode", "ai_manager")
        add_flag(command, "async_manager_lead_updates", manager["async_lead_updates"])
        add_flag(command, "async_manager_max_wait_updates", manager["async_max_wait_updates"])
        add_flag(command, "async_manager_wait_timeout_seconds", manager["async_wait_timeout_seconds"])
        add_flag(command, "llm_model", manager["reasoning_model"])
        add_flag(command, "llm_temperature", manager["reasoning_temperature"])
        add_flag(command, "llm_formatter_model", manager["formatter_model"])
        add_flag(command, "llm_formatter_temperature", manager["formatter_temperature"])
        add_flag(command, "rollback_min_drop", manager["rollback_min_drop"])
        add_flag(command, "rollback_patience", manager["rollback_patience"])
        add_flag(command, "rollback_cooldown_intervals", manager["rollback_cooldown_intervals"])
    else:
        add_flag(command, "mode", "fixed")

    return command, out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an exact released randomized-reacher configuration.")
    parser.add_argument("--condition", choices=["baseline", "conservative", "aggressive"], required=True)
    parser.add_argument(
        "--seed",
        type=nonnegative_seed,
        required=True,
        help="Training seed. Any non-negative integer may be used; the paper reports seeds 40, 41, and 42.",
    )
    parser.add_argument("--manager", action="store_true", help="Enable the two-stage asynchronous manager.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command, out_dir = build_command(args)
    print("Resolved command:")
    print(" ".join(command))
    print(f"Output directory: {out_dir}")
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
