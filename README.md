# Bounded AI Training Manager for Randomized Reacher PPO

Anjali Rao and Nikhil Kamalkumar Advani

Code for the paper [AI Training Manager: Bounded Closed-Loop Control of Adaptive
Training Recipes](https://arxiv.org/abs/2606.29871) (arXiv:2606.29871).

This repository contains the released reinforcement-learning environment,
training code, two-stage AI training manager, exact experiment configurations,
action bounds, response verifier, random seeds, and reproduction instructions
used for the randomized robotic-arm experiments.

The repository contains source and reproducibility artifacts only. It excludes
generated run outputs, private API keys, and cached model responses.

## Task

The environment is a two-link planar arm with two continuous joint-velocity
actions. Every episode independently samples:

- the initial joint configuration;
- the target position;
- the circular obstacle position; and
- the obstacle radius.

An episode is successful only when the end effector reaches the target without
collision. A collision terminates the episode. Training scenes are sampled from
a mixture of easy, medium, and hard geometries; evaluation reports each split
separately.

The observation has 14 values: sine and cosine of both joint angles, target
position, obstacle position and radius, end-effector position, target vector,
and obstacle clearance. Actions are clipped to `[-1, 1]` before application.

### Scene Sampling

The start angles are sampled uniformly from `[-0.9, 0.9]` and `[-1.1, 1.1]`.
Targets use a radius in `[0.40, 0.95]` and polar angle in `[0.25, 2.65]`, with
the initial end-effector-to-target distance restricted to `[0.25, 0.95]`.
Obstacles are placed 35%-70% of the way along the start-to-target line, on a
random side of that line.

Training episodes use this mixture:

| Mixture weight | Direct-path clearance | Obstacle radius |
|---:|---:|---:|
| 45% | `[-0.025, 0.06]` | `[0.075, 0.115]` |
| 35% | `[0.04, 0.14]` | `[0.06, 0.10]` |
| 20% | `[0.14, 0.28]` | `[0.045, 0.075]` |

The fixed diagnostic and held-out splits use:

| Split | Direct-path clearance | Obstacle radius |
|---|---:|---:|
| easy | `[0.12, 0.28]` | `[0.04, 0.07]` |
| medium | `[0.02, 0.13]` | `[0.06, 0.10]` |
| hard | `[-0.03, 0.06]` | `[0.075, 0.12]` |

Invalid scenes are rejected when the arm begins too close to the obstacle, the
target is insufficiently clear of it, or the sampled geometry lies outside the
workspace constraints in `env.py`.

### Reward

The per-step reward is:

```text
progress_scale * (previous_distance - current_distance)
+ success_bonus * safe_success
- collision_penalty * collision
- 0.01 * ||action||^2
- 0.05 * normalized_clearance_shortfall
```

The manager may tune the first three named weights, but it cannot change the
definition of collision, safe success, termination, or the scene distribution.

## Repository Contents

- `env.py`: randomized Gymnasium environment and reward calculation.
- `train_ppo.py`: PyTorch PPO trainer, telemetry, evaluation, checkpointing,
  asynchronous manager integration, and held-out final evaluation.
- `manager.py`: action surface, prompt construction, two-stage OpenAI calls,
  strict JSON/action verification, and recipe updates.
- `configs/`: exact common protocol and baseline, conservative, and aggressive
  recipes.
- `run_experiment.py`: resolves a released config into a trainer command.
- `export_prompts.py`: renders both exact prompt stages without an API call.
- `prompts/`: checked-in rendered prompt examples.
- `tests/test_manager.py`: prompt, verifier, action-surface, checkpoint, and
  rollback tests.
- `verify_release.py`: one-command release/configuration verification.
- `plot.py`: per-run diagnostic plots and sampled task geometry.
- `LICENSE`: MIT license.

## Installation

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_release.py
```

`--device auto` uses Apple MPS when available and otherwise uses CPU. To require
MPS, pass `--device mps`; the trainer fails immediately if MPS is unavailable.

## Released Recipes

| Parameter | Baseline | Conservative | Aggressive |
|---|---:|---:|---:|
| actor learning rate | `3e-4` | `3e-5` | `3e-3` |
| critic learning rate | `1e-3` | `1e-3` | `3e-3` |
| entropy coefficient | `0.01` | `0.001` | `0.08` |
| PPO clip coefficient | `0.20` | `0.10` | `0.30` |
| PPO update epochs | `6` | `3` | `8` |
| target KL | `0.03` | `0.01` | `0.10` |
| initial actor log standard deviation | `-0.7` | `-0.7` | `-0.7` |
| collision penalty | `5` | `25` | `1` |
| safe-success bonus | `5` | `2.5` | `10` |
| progress scale | `3` | `1.5` | `5` |

Shared settings are `16` environments, `128` steps per PPO update, minibatches
of `256`, `120` maximum episode steps, `500` PPO updates (1,024,000 environment
steps), success radius `0.09`, and `80` diagnostic episodes per difficulty and
evaluation mode. Fixed runs evaluate every 10 updates. Manager runs use a
20-update intervention interval.

Other fixed PPO settings are discount factor `0.99`, GAE lambda `0.95`, value
loss coefficient `0.5`, gradient-norm clipping at `0.5`, and no learning-rate
annealing. Fixed environment terms are action penalty `0.01`, clearance penalty
`0.05`, and clearance margin `0.08`; link lengths are `0.65` and `0.55`, and the
simulation time step is `0.08`.

The machine-readable source of truth is `configs/`. `run_experiment.py` reads
those files directly; the table above is explanatory, not a second config.

## Reproducing Runs

The results reported in the [paper](https://arxiv.org/abs/2606.29871) use seeds `40`, `41`, and `42`. These are
recorded for exact reproduction; `--seed` accepts any non-negative integer, so
users may run additional seeds or their own seed sets.

Run a fixed baseline:

```bash
python run_experiment.py --condition baseline --seed 40 --device auto
```

Run either fixed comparison condition:

```bash
python run_experiment.py --condition conservative --seed 40 --device auto
python run_experiment.py --condition aggressive --seed 40 --device auto
```

Run a condition with the manager:

```bash
export OPENAI_API_KEY='your-key'
python run_experiment.py --condition conservative --seed 40 --manager --device auto
python run_experiment.py --condition aggressive --seed 40 --manager --device auto
```

Manager runs make paid OpenAI API calls. A 500-update run requests about 25
decisions, each consisting of one `gpt-5.4` call and one or two `gpt-5.4-mini`
calls.

Use seeds `41` and `42` to reproduce the other reported runs, or provide any
other non-negative integer seed for additional experiments. Runs are
intentionally launched one at a time; this keeps GPU/MPS contention and API
accounting explicit. Use `--dry-run` to print the fully resolved trainer command
without training or calling the API.

## Evaluation Protocol and Seeds

During training, the policy is evaluated on a fixed diagnostic scene set in
both deterministic and stochastic modes. These diagnostics are visible to the
manager and participate in checkpoint selection, so they are not the final test
set.

For training seed `S`, the diagnostic seed base is:

```text
S * 1,000,000 + 100,000
```

After training, the frozen final policy and selected best checkpoint are
evaluated on a separate held-out scene set that was never shown to the manager.
For training seed `S`, its seed base is:

```text
S * 10,000,000 + 9,000,000
```

The held-out evaluation uses 240 episodes per difficulty in each deterministic
and stochastic mode. Its outputs are `final_heldout_eval.json` and
`final_heldout_eval_by_difficulty.csv`.

## Manager Protocol

Manager runs use two calls:

1. Stage 1: `gpt-5.4`, temperature `0.6`. It receives task guidance, the current
   recipe, compact raw telemetry, factual prior action outcomes, and only the
   actions executable at that call. It makes the substantive decision in plain
   English.
2. Stage 2: `gpt-5.4-mini`, temperature `0.1`. It receives only Stage 1's text,
   the current action surface, the output schema, and mechanical validation
   feedback on one retry. It formats the decision as JSON and is instructed not
   to re-diagnose the run.

Calls are asynchronous. For a 20-update intervention interval, requests are
launched five updates early at updates `15, 35, 55, ...`. A decision that
arrives within five updates of its request is verified against the current
trainer state and applied at the intended boundary. Training never pauses for
the manager: a decision that has not arrived within five updates is discarded as
stale, logged as `ASYNC_MANAGER_STALE`, and the current recipe continues
unchanged.

Every successful manager record stores the exact Stage 1 prompt, Stage 1 output,
each Stage 2 prompt and raw response, validation errors, selected action, and
recipe after application in `controller_decisions.jsonl`.

### Exact Prompts

`manager.py::build_analysis_prompt` and `manager.py::build_prompt` are the
executable prompt definitions. The prompt necessarily contains dynamic values,
so a single static file cannot represent every call. To inspect the exact
rendering without contacting OpenAI:

```bash
python export_prompts.py --condition aggressive --seed 40
```

This writes `prompts/stage1_example.txt` and `prompts/stage2_example.txt`. Real
run logs preserve the exact rendered prompt used for every intervention.

## Action Surface and Bounds

The manager can change only the controls below. Choices that would have no
effect after clipping are removed before the action surface is shown.

| Control | Permitted values per intervention | Absolute bound |
|---|---|---|
| actor LR | multiply by `0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2, 3, 5` | `[1e-5, 3e-3]` |
| critic LR | multiply by `0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2, 3` | `[1e-5, 3e-3]` |
| entropy coefficient | multiply by `0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2, 3, 5` | `[0, 0.1]` |
| clip coefficient | set to `0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30` | `[0.03, 0.30]` |
| PPO epochs | set to `1, 2, 3, 4, 6, 8` | `[1, 8]` |
| target KL | set to `0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08` | `[0.003, 0.10]` |
| collision penalty | set to `1, 2, 5, 10, 20, 35, 50` | `[1, 50]` |
| safe-success bonus | set to `2.5, 3, 5, 8, 10` | `[2.5, 10]` |
| progress scale | set to `1, 2, 3, 5` | `[1, 5]` |

The structural actions are `no_action`, `rollback_to_best`,
`rollback_and_consolidate`, `stop_training`, and `combined_policy`.
`combined_policy` must contain exactly two or three available controls.
`rollback_and_consolidate` restores a checkpoint and applies exactly one
available conservative change.

The learned actor log standard deviation is telemetry, not a manager control.
The manager cannot alter environment geometry, collision termination, success
criteria, success radius, architecture, optimizer type, or evaluation protocol.

## Verifier and Rollback Gate

Stage 2 output is not executed directly. `manager.py::validate_response` checks:

- required JSON fields and types;
- that the selected action is available on the current call;
- exact control names and permitted values;
- combined-action cardinality;
- rollback and stopping eligibility; and
- that the bounded update changes the recipe.

The formatter receives one retry with only the mechanical validation error. If
it still fails, the action becomes `no_action` and the failure is logged.

The best checkpoint is selected using both deterministic and stochastic
evaluation across easy, medium, and hard scenes. Its robust score is the worst
safe-success cell minus the worst collision cell. Rollback becomes available
only after the current robust score falls at least `0.10` below the best score
for two consecutive evaluations and the two-evaluation cooldown has elapsed.
Rollback restores policy and optimizer state while retaining the active manager
recipe. Stopping is offered only after two consecutive evaluations in which all
six mode-by-difficulty cells have safe success at least `0.85` and collision at
most `0.15`.

Run the release verifier with:

```bash
python verify_release.py
```

## Outputs

Each run directory contains:

- `run_config.json`: fully resolved command-line configuration;
- `metrics.csv`: evaluation and PPO-health metrics;
- `updates.jsonl`: one PPO-health record for every update;
- `telemetry.jsonl`: full structured telemetry;
- `controller_decisions.jsonl`: manager requests, prompts, outputs, actions,
  and outcomes for manager runs;
- `best.pt` and `final.pt`: PyTorch checkpoints;
- `final_heldout_eval.json` and `final_heldout_eval_by_difficulty.csv`.

Generate per-run plots with:

```bash
python plot.py --run_dir runs/aggressive_manager_seed40
```

## Citation

If you use this code, please cite:

```bibtex
@misc{rao2026aitrainingmanagerbounded,
      title={AI Training Manager: Bounded Closed-Loop Control of Adaptive Training Recipes},
      author={Anjali Rao and Nikhil Kamalkumar Advani},
      year={2026},
      eprint={2606.29871},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.29871},
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
