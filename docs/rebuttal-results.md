# Additional Training Results

This page reports additional supervised fine-tuning (SFT) results and PPO
training diagnostics for Agentick.

## Supervised fine-tuning

We LoRA-fine-tune Qwen3.5-4B on per-step state-action examples from the
Agentick oracle trajectory datasets. The 120k and 250k configurations use the
same ASCII observation interface and are evaluated on the same 37 tasks,
four difficulties, and 25 evaluation seeds per task-difficulty pair as the
corresponding base-model results.

| Harness | Base Qwen3.5-4B | SFT-120k | SFT-250k |
|---|---:|---:|---:|
| Markovian | 0.0232 | 0.3544 | **0.4466** |
| Markovian reasoner | 0.2280 | 0.3489 | **0.4436** |

![SFT training curves and evaluation results](assets/rebuttal/sft_results_combined.png)

## Difficulty scaling validation

Difficulty labels select parameter regimes of each seeded procedural generator;
they do not select hand-authored levels. Every seed generates a new instance,
while easy through expert jointly scale task-relevant axes such as grid size,
object or constraint count, stochasticity, and horizon. The axes and four
parameter presets are benchmark design choices.

The compact table reports capability-balanced mean success across all 37 tasks
and 25 official seeds per task-difficulty pair.

| Agent | Easy | Medium | Hard | Expert | Easy→Expert drop |
|---|---:|---:|---:|---:|---:|
| GPT-5 mini | .494 | .332 | .223 | .189 | 61.8% |
| PPO dense 2M | .606 | .277 | .160 | .104 | 82.8% |
| Qwen3.5-4B | .445 | .259 | .110 | .098 | 78.0% |
| SFT-250k | .647 | .467 | .369 | .303 | 53.1% |
| **Mean** | **.548** | **.334** | **.215** | **.173** | **68.3%** |

Every agent degrades strictly from easy to expert. Averaged across four
distinct training/evaluation regimes, success falls from .548 to .173 (68.3%).

## PPO learning curves

PPO is trained independently for every task-difficulty pair: the benchmark
contains 148 separate policies rather than one multitask policy. The figures
below show the complete 2M-step trajectory for every available policy. Each
subplot contains the easy, medium, hard, and expert policies for one task.
Color denotes difficulty. Every trajectory is continuous: the line is solid
from 500k to 2M steps and dashed before 500k, with a marker at the 500k
checkpoint. The y-axis is the periodic evaluation success rate, smoothed with
a 31-point trailing rolling mean so the 500k marker uses no post-500k data.

| Recovered policies | 500k | 1M | 1.5M | 2M | Δ 1.5M→2M |
|---|---:|---:|---:|---:|---:|
| Easy (37) | .441 | .573 | .656 | .679 | +.024 |
| Medium (37) | .217 | .282 | .313 | .339 | +.026 |
| Hard (31) | .167 | .221 | .243 | .239 | −.003 |
| Expert (21) | .141 | .181 | .209 | .209 | +.000 |
| **Task-balanced mean** | **.243** | **.329** | **.377** | **.393** | **+.016** |

The successive task-balanced gains shrink from +.086 to +.048 to +.016. From
1.5M to 2M, the median task-level absolute change is .016; 32/37 task
aggregates change by at most .05 and 36/37 by at most .10. At 2M, 17/21 tasks
with all four recovered curves order easy ≥ medium ≥ hard ≥ expert. These are
smoothed deterministic diagnostic episodes; final leaderboard evaluation uses
25 official seeds.

### Navigation

![PPO navigation task learning curves](assets/rebuttal/ppo_navigation_task_learning_curves.png)

### Planning

![PPO planning task learning curves](assets/rebuttal/ppo_planning_task_learning_curves.png)

### Reasoning

![PPO reasoning task learning curves](assets/rebuttal/ppo_reasoning_task_learning_curves.png)

### Memory

![PPO memory task learning curves](assets/rebuttal/ppo_memory_task_learning_curves.png)

### Generalization

![PPO generalization task learning curves](assets/rebuttal/ppo_generalization_task_learning_curves.png)

### Multi-agent

![PPO multi-agent task learning curves](assets/rebuttal/ppo_multi_agent_task_learning_curves.png)
