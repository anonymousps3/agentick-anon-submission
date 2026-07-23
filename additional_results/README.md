# Additional results

This directory collects the figures and compact tables discussed in the author
response.

**No PPO or SFT model was retrained, and no evaluation was rerun specifically
for the response.** These artifacts summarize results that already existed
either in the leaderboard or in saved local training/evaluation outputs but
were not fully presented in the submitted paper.

## Provenance

- **PPO learning curves:** derived from the saved periodic logs of the same
  148 independent task-difficulty policies whose 500k/2M evaluations were
  already reported. Logs were available for 126 policies. The response exposes
  these existing traces and checkpoint aggregates; it does not add PPO
  training.
- **SFT:** derived from the existing Qwen3.5-4B LoRA training logs, checkpoints,
  and stored 37-task evaluation results. The response summarizes these existing
  artifacts; it does not add SFT training or evaluation.
- **Difficulty scaling:** aggregated from existing per-agent, per-task, and
  per-difficulty leaderboard results over the official 25 evaluation seeds. No
  environment evaluation was rerun.

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

| Harness | Base | SFT-120k | SFT-250k |
|---|---:|---:|---:|
| Markovian | .0232 | .3544 | **.4466** |
| Markovian reasoner | .2280 | .3489 | **.4436** |

### PPO saved training traces

| Recovered traces | 500k | 1M | 1.5M | 2M | Δ 1.5M→2M |
|---|---:|---:|---:|---:|---:|
| Easy (37) | .441 | .573 | .656 | .679 | +.024 |
| Medium (37) | .217 | .282 | .313 | .339 | +.026 |
| Hard (31) | .167 | .221 | .243 | .239 | -.003 |
| Expert (21) | .141 | .181 | .209 | .209 | +.000 |
| **Task-balanced mean** | **.243** | **.329** | **.377** | **.393** | **+.016** |

The curves are periodic deterministic diagnostic episodes, smoothed with a
31-point trailing mean. Lines are dashed before 500k, marked at 500k, and solid
thereafter. Final leaderboard evaluations use all 25 official seeds.

### Difficulty scaling

| Agent | Easy | Medium | Hard | Expert | Easy→Expert drop |
|---|---:|---:|---:|---:|---:|
| GPT-5 mini | .494 | .332 | .223 | .189 | 61.8% |
| PPO dense 2M | .606 | .277 | .160 | .104 | 82.8% |
| Qwen3.5-4B | .445 | .259 | .110 | .098 | 78.0% |
| SFT-250k | .647 | .467 | .369 | .303 | 53.1% |
| **Mean** | **.548** | **.334** | **.215** | **.173** | **68.3%** |
