# 🕹️ Deep Q-Learning for Atari Amidar

A full Deep Q-Network (DQN) implementation trained on the **ALE/Amidar-v5** Atari environment, built as a course assignment. Includes baseline training, hyperparameter ablations, exploration strategy comparisons, and connections to LLM-based agents.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Environment](#environment)
- [Architecture & Algorithm](#architecture--algorithm)
- [Exploration Strategies](#exploration-strategies)
- [Results](#results)
- [Ablations](#ablations)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Attribution](#attribution)
- [License](#license)

---

## Overview

This project trains a DQN agent to play *Amidar*, a classic Atari game in which the player traces paths on a grid while avoiding enemies. The agent learns end-to-end from raw pixel frames using a convolutional neural network.

**Key techniques implemented:**
- Experience Replay — random minibatches to break temporal correlations
- Target Network — a slower-updating copy of the Q-network to stabilize bootstrapping
- Reward Clipping — bounding rewards to `[-1, 1]` for gradient stability
- ε-greedy and Boltzmann exploration strategies
- Double DQN target logic

**Baseline results (300 episodes):**

| Policy | Avg Return | Best Episode |
|---|---|---|
| Random (10 eps) | 0.80 | — |
| DQN baseline | 5.01 (σ=5.24) | 24.00 |

---

## Environment

**`ALE/Amidar-v5`** via [Gymnasium](https://gymnasium.farama.org/) + ALE-py.

| Property | Value |
|---|---|
| Observation space | `Box(0, 255, (210, 160, 3), uint8)` |
| Action space | `Discrete(10)` |
| Preprocessing | Grayscale → 84×84, stacked 4 frames |
| Why no Q-table? | `\|S\| ≈ 256^(210×160×3)`, `\|A\| = 10` → intractable |

Stacking 4 grayscale frames gives the agent motion cues (velocity/direction of enemies and the player) that a single frame cannot provide.

---

## Architecture & Algorithm

The agent uses an "Atari-style" CNN head mapping stacked pixel frames to action-values Q(s, a).

**Bellman target:**

$$y = r + \gamma \max_{a'} Q_{\bar\theta}(s', a')$$

**Loss:** Huber loss applied to the TD error.

**Q-learning update:**

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma \max_{a'} Q_{\bar\theta}(s',a') - Q(s,a)\Big]$$

**Baseline hyperparameters:**

| Param | Value |
|---|---|
| Learning rate α | 2.5e-4 |
| Discount γ | 0.99 |
| Target update | Polyak (soft) updates |
| Gradient clipping | Yes |

---

## Exploration Strategies

Two exploration strategies are compared:

**ε-greedy (baseline):**
- ε decays `1.0 → 0.05` over **300,000** steps
- Extended schedule study: `1.0 → 0.01` over **600,000** steps

**Boltzmann (softmax):**

$$p(a \mid s) \propto \exp\!\left(\frac{Q(s,a)}{\tau}\right)$$

A higher temperature τ spreads probability more evenly across actions (more exploratory); lower τ concentrates on the greedy action.

---

## Results

Training metrics are logged per episode and saved as JSON files. Evaluation checkpoints are run every 50 episodes.

**Baseline (α=2.5e-4, γ=0.99) eval scores:**

| Episode | Avg Eval Return |
|---|---|
| 50 | 3.0 |
| 100 | 6.2 |
| 150 | 7.4 |
| 200 | 15.4 |
| 250 | 19.8 |

Metrics tracked per run: average return, average steps/episode, loss curve, epsilon schedule, and per-episode evaluation scores.

---

## Ablations

Three ablation conditions are included alongside the baseline:

| Run | α | γ | Notes |
|---|---|---|---|
| `amidar_metrics.json` | 2.5e-4 | 0.99 | Baseline |
| `amidar_metrics_a1e-4_g0_95.json` | 1e-4 | 0.95 | Lower LR, shorter horizon |
| `amidar_metrics_a2_5e-4_g0_99.json` | 2.5e-4 | 0.99 | Baseline rerun |
| `amidar_metrics_boltz_tau0_7.json` | 2.5e-4 | 0.99 | Boltzmann τ=0.7 |
| `amidar_metrics_eps_1_0_to_0_01_over_600k.json` | 2.5e-4 | 0.99 | Extended ε decay |

The lower learning rate / shorter discount (`α=1e-4, γ=0.95`) tends to be more stable early in training but achieves a lower peak return compared to the baseline.

---

## Installation

This project runs on **Google Colab** (T4 GPU recommended). Install dependencies with:

```bash
pip -q install "gymnasium[atari,accept-rom-license]==1.2.1" ale-py==0.10.1 autorom==0.6.1 \
               torch torchvision tqdm matplotlib opencv-python imageio pandas
AutoROM --accept-license
```

**Key package versions:**

| Package | Version |
|---|---|
| gymnasium[atari] | 1.2.1 |
| ale-py | 0.10.1 |
| autorom | 0.6.1 |
| torch | Colab default (CUDA) |

---

## Usage

Open `DQN_Amidar_Assignment.ipynb` in Google Colab and run cells sequentially. The notebook covers:

1. Environment setup and preprocessing
2. DQN model definition (CNN + replay buffer)
3. Baseline training (300 episodes)
4. Evaluation and metric export
5. Ablation runs (α/γ, exploration schedule, Boltzmann)
6. Analysis, plots, and rubric answers

Metrics are automatically exported to JSON files after each training run.

---

## Project Structure

```
├── DQN_Amidar_Assignment.ipynb         # Main notebook
├── AMIDAR_AUTO_ANSWERS.md              # Rubric Q&A (auto-generated)
├── amidar_random_baseline.json         # Random policy baseline metrics
├── amidar_metrics.json                 # Baseline DQN training metrics
├── amidar_metrics_a1e-4_g0_95.json     # Ablation: α=1e-4, γ=0.95
├── amidar_metrics_a2_5e-4_g0_99.json   # Ablation: α=2.5e-4, γ=0.99 (rerun)
├── amidar_metrics_boltz_tau0_7.json    # Ablation: Boltzmann τ=0.7
├── amidar_metrics_eps_1_0_to_0_01_over_600k.json  # Ablation: long ε decay
└── README.md
```

---

## DQN vs LLM-Based Agents

| Aspect | DQN (RL) | LLM Agents |
|---|---|---|
| **Signal** | Dense step-wise numeric reward | Human/preference reward (RLHF/RLAIF), often sparse |
| **Policy** | Value-based: act w.r.t. Q(s,a) | Emergent from generative modeling + reasoning |
| **Goal** | Maximize expected return | Alignment, instruction following, tool use |
| **Planning** | Value iteration, rollouts, MCTS | Chain/tree-of-thought + tool calls |

The value estimation and exploration ideas from DQN transfer conceptually to LLM agent settings — e.g., a Planner–Controller architecture where an LLM sets subgoals and a DQN-style policy executes low-level actions.

---

## Attribution

**Original work:** DQN agent implementation (CNN model, replay buffer, ε/Boltzmann exploration, Double DQN target logic, training/eval loops), Amidar preprocessing, checkpointing, metrics export, ablation runners, and documentation.


**Third-party libraries:**

| Package | License |
|---|---|
| Gymnasium / ALE-py / AutoROM | MIT / BSD |
| PyTorch | BSD-style |
| NumPy | BSD-3 |
| OpenCV-python | Apache-2.0 |
| ImageIO | BSD-2 |
| TQDM | MPL-2.0 |

*Amidar* gameplay recordings are produced via the ALE academic interface for educational purposes.

---

## License

MIT License — Copyright (c) 2026 Husain Yusuf

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
