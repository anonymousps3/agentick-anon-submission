# Additional results

This directory contains compact, machine-readable results and figures added for
the author response: supervised fine-tuning, PPO observation ablations, PPO
learning diagnostics, and difficulty scaling.

## PPO observation ablation

We repeat the single-task dense-reward PPO protocol with four synchronized
observation paths. Each task-difficulty pair uses a separate policy. Task
instances, reward, PPO optimization hyperparameters, 2M environment-step
budget, evaluation cadence, checkpoint cadence, and deterministic 25-episode
final-evaluation protocol are held fixed. Every input uses four consecutive
observations.

Only the observation-compatible input path changes:

- pixels: 84x84 grayscale with a CNN;
- exposed grid state: the returned `terrain`, `objects`, `agents`, and
  `metadata` grid layers, flattened for an MLP;
- ASCII: padded byte IDs with a learned byte embedding and 2D CNN;
- natural language: padded byte IDs with a learned byte embedding and 1D CNN.

The grid vector uses only visibility-filtered arrays returned by the
`state_dict` observation. It excludes hidden internal cells, task
configuration, annotations, and valid-action masks. The text encoders are
trained from scratch and add no pretrained language knowledge. Because the
compatible neural encoder necessarily changes, this comparison measures each
declared observation-plus-encoder setup rather than pure information content
in isolation.

### Exact final success by category

| Observation | Overall | Navigation | Planning | Reasoning | Memory | Generalization | Multi-agent |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO, pixels (CNN, dense, 2M) | .196 | .225 | .294 | .010 | .280 | .167 | .222 |
| PPO, grid-state vector (MLP, dense, 2M) | .102 | .110 | .064 | .086 | .083 | .113 | .194 |
| PPO, ASCII (byte CNN, dense, 2M) | .192 | .211 | .270 | .085 | .233 | .133 | .194 |
| PPO, natural language (byte CNN, dense, 2M) | .188 | .199 | .303 | .034 | .250 | .120 | .204 |

### Exact final success by difficulty

| Observation | Easy | Medium | Hard | Expert |
|---|---:|---:|---:|---:|
| PPO, pixels (CNN, dense, 2M) | .486 | .176 | .078 | .044 |
| PPO, grid-state vector (MLP, dense, 2M) | .183 | .121 | .070 | .036 |
| PPO, ASCII (byte CNN, dense, 2M) | .394 | .211 | .107 | .056 |
| PPO, natural language (byte CNN, dense, 2M) | .378 | .219 | .072 | .083 |

The tables use cell-balanced final success. Every modality contains all 148
task-difficulty cells. The three new modalities store the exact 25 boolean
success flags, returns, and episode lengths for every cell; the historical
pixel run stores the exact aggregate computed from `info["success"]`. Dense
return is never converted into final success.

Machine-readable sources:
[category table](ppo_observation_ablation_by_category.csv) and
[difficulty table](ppo_observation_ablation_by_difficulty.csv).

The result is deliberately nuanced. Pixels, ASCII, and natural language are
close overall, while the plain grid-vector MLP is weaker. Category rankings
still change materially: natural language is strongest on the planning group,
the grid vector and ASCII substantially improve over pixels on the reasoning
group, and pixels remain strongest on navigation and memory. Observation and
compatible encoder therefore matter; the broader cross-paradigm table should
not be read as an input-independent ranking.

## PPO learning diagnostics

The periodic evaluator retained episode returns but not exact success flags.
Consequently, the curves and checkpoint table below report the **positive-return
episode fraction**, not exact success. They use the exact intersection of 126
available task-difficulty traces for all four modalities, a 31-point trailing
mean, and a fixed 0–1 vertical scale.

| Observation | 500k | 1M | 1.5M | 2M |
|---|---:|---:|---:|---:|
| PPO, pixels (CNN, dense, 2M) | .243 | .329 | .377 | .393 |
| PPO, grid-state vector (MLP, dense, 2M) | .193 | .208 | .214 | .220 |
| PPO, ASCII (byte CNN, dense, 2M) | .162 | .303 | .336 | .369 |
| PPO, natural language (byte CNN, dense, 2M) | .269 | .342 | .364 | .364 |

The checkpoint average follows the original pixel convention: difficulties
available for a task are averaged first, then the 37 task means are averaged
equally. Source tables:
[compact checkpoints](ppo_observation_positive_return_by_checkpoint.csv) and
[difficulty-detail checkpoints](ppo_observation_positive_return_checkpoint_summary.csv).

Matched category-curve grids:

- Pixels:
  [navigation](ppo_navigation_task_learning_curves.png),
  [planning](ppo_planning_task_learning_curves.png),
  [reasoning](ppo_reasoning_task_learning_curves.png),
  [memory](ppo_memory_task_learning_curves.png),
  [generalization](ppo_generalization_task_learning_curves.png), and
  [multi-agent](ppo_multi_agent_task_learning_curves.png).
- Grid-state vector:
  [navigation](ppo_vector_navigation_matched_task_positive_return_curves.png),
  [planning](ppo_vector_planning_matched_task_positive_return_curves.png),
  [reasoning](ppo_vector_reasoning_matched_task_positive_return_curves.png),
  [memory](ppo_vector_memory_matched_task_positive_return_curves.png),
  [generalization](ppo_vector_generalization_matched_task_positive_return_curves.png), and
  [multi-agent](ppo_vector_multi_agent_matched_task_positive_return_curves.png).
- ASCII:
  [navigation](ppo_ascii_navigation_matched_task_positive_return_curves.png),
  [planning](ppo_ascii_planning_matched_task_positive_return_curves.png),
  [reasoning](ppo_ascii_reasoning_matched_task_positive_return_curves.png),
  [memory](ppo_ascii_memory_matched_task_positive_return_curves.png),
  [generalization](ppo_ascii_generalization_matched_task_positive_return_curves.png), and
  [multi-agent](ppo_ascii_multi_agent_matched_task_positive_return_curves.png).
- Natural language:
  [navigation](ppo_language_navigation_matched_task_positive_return_curves.png),
  [planning](ppo_language_planning_matched_task_positive_return_curves.png),
  [reasoning](ppo_language_reasoning_matched_task_positive_return_curves.png),
  [memory](ppo_language_memory_matched_task_positive_return_curves.png),
  [generalization](ppo_language_generalization_matched_task_positive_return_curves.png), and
  [multi-agent](ppo_language_multi_agent_matched_task_positive_return_curves.png).

## SFT evaluation

| Prompting method | Base | SFT-120k | SFT-250k |
|---|---:|---:|---:|
| Direct action | .0232 | .3544 | **.4466** |
| Reason before acting | .2280 | .3489 | **.4436** |

See the [combined SFT figure](sft_results_combined.png) and
[source table](sft_evaluation_summary.csv).

## Difficulty scaling

| Agent | Easy | Medium | Hard | Expert | Easy→Expert drop |
|---|---:|---:|---:|---:|---:|
| GPT-5 mini | .494 | .332 | .223 | .189 | 61.8% |
| PPO dense 2M | .486 | .176 | .078 | .044 | 90.9% |
| Qwen3.5-4B | .445 | .259 | .110 | .098 | 78.0% |
| SFT-250k | .647 | .467 | .369 | .303 | 53.1% |
| **Average across agents** | **.518** | **.309** | **.195** | **.159** | **69.4%** |

These PPO entries use the same exact final-success aggregate as the observation
ablation. See the [source table](difficulty_scaling_summary.csv).
