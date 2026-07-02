# Oil Well Production Optimization using Q-Learning

This project uses Reinforcement Learning (Q-Learning) to find the optimal production control strategy for an oil well.

## Objective

Train an agent that maximizes oil production while avoiding operational risks like high water cut and low reservoir pressure.

## How It Works

### States (4)
| State | Description |
|---|---|
| Low Production | Output is below target |
| Normal Production | Output is at desired level |
| High Water Cut | Too much water in production |
| Low Reservoir Pressure | Pressure is too low |

### Actions (3)
| Action | Description |
|---|---|
| Increase Rate | Boost production |
| Decrease Rate | Reduce production |
| Maintain Rate | Keep current rate |

### Rewards
| Situation | Best Action | Reward |
|---|---|---|
| Low Production | Increase | +20 |
| Normal Production | Maintain | +15 |
| High Water Cut | Decrease | +10 |
| Low Reservoir Pressure | Decrease | +20 |
| Wrong action | Any other | -20 |

## Algorithm

- **Q-Learning** with ε-greedy policy
- Learning rate: 0.1 | Discount factor: 0.9 | Exploration: 0.1
- Trained for 1000 episodes

## Learned Policy

| State | Best Action |
|---|---|
| Low Production | Increase Rate |
| Normal Production | Maintain Rate |
| High Water Cut | Decrease Rate |
| Low Reservoir Pressure | Decrease Rate |

## Files

- `reinforcement.ipynb` — Jupyter notebook with full implementation, training, evaluation, and visualizations

## Requirements

- Python 3
- numpy, pandas, matplotlib, seaborn, jupyter
