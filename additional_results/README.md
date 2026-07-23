# Additional results

This directory collects the figures and compact tables discussed in the author
response: SFT training and evaluation, PPO learning curves, checkpoint
aggregates, and difficulty-scaling results.

## Files

- [SFT training and evaluation](sft_results_combined.png).
- PPO task-grid learning curves:
  [navigation](ppo_navigation_task_learning_curves.png),
  [planning](ppo_planning_task_learning_curves.png),
  [reasoning](ppo_reasoning_task_learning_curves.png),
  [memory](ppo_memory_task_learning_curves.png),
  [generalization](ppo_generalization_task_learning_curves.png), and
  [multi-agent](ppo_multi_agent_task_learning_curves.png).
- Source tables:
  [SFT evaluation](sft_evaluation_summary.csv),
  [PPO checkpoints](ppo_checkpoint_summary.csv), and
  [difficulty scaling](difficulty_scaling_summary.csv).

## Compact tables

### SFT evaluation

| Prompting method | Base | SFT-120k | SFT-250k |
|---|---:|---:|---:|
| Direct action | .0232 | .3544 | **.4466** |
| Reason before acting | .2280 | .3489 | **.4436** |

### PPO saved training traces

| Checkpoint | Easy | Medium | Hard | Expert | Average | Gain since prior |
|---|---:|---:|---:|---:|---:|---:|
| 500k | .441 | .217 | .167 | .141 | .243 | — |
| 1M | .573 | .282 | .221 | .181 | .329 | +.086 |
| 1.5M | .656 | .313 | .243 | .209 | .377 | +.048 |
| 2M | .679 | .339 | .239 | .209 | .393 | +.016 |

The curves evaluate the same fixed test episode every 10k training steps and
use a 31-point trailing average for smoothing. Lines are dashed before 500k,
marked at 500k, and solid thereafter. The final table column is the change in
the overall average since the previous checkpoint. It shrinks from +.086 to
+.048 to +.016, so the final 1.5M→2M interval is the smallest. Final leaderboard
evaluations use all official seeds. The six category-level learning-curve
figures are available directly in this directory.

### Difficulty scaling

| Agent | Easy | Medium | Hard | Expert | Easy→Expert drop |
|---|---:|---:|---:|---:|---:|
| GPT-5 mini | .494 | .332 | .223 | .189 | 61.8% |
| PPO dense 2M | .606 | .277 | .160 | .104 | 82.8% |
| Qwen3.5-4B | .445 | .259 | .110 | .098 | 78.0% |
| SFT-250k | .647 | .467 | .369 | .303 | 53.1% |
| **Average across agents** | **.548** | **.334** | **.215** | **.173** | **68.3%** |
