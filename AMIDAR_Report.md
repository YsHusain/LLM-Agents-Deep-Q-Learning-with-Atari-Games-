
# Amidar DQN — Rubric Answers (Auto)

## Baseline Performance
Random policy (10 eps, max_steps=1000): avg_return=0.8, avg_steps=581.1
DQN (300 eps): avg_return=5.01 (σ=5.24), best=24.00

## Environment Analysis
Observations: Box(0, 255, (210, 160, 3), uint8) → grayscale 84×84 × stack 4.
Actions: Discrete(10). Q-table size: ~infeasible (|S|≈256^(210*160*3), |A|=10).

## Reward Structure
Native score each step; optional reward clipping [-1,1] for stability.

## Bellman Parameters
α=0.00025, γ=0.99; ablations for α=1e-4, γ=0.95 logged.

## Policy Exploration
Baseline ε-greedy; alternative Boltzmann (τ).

## Exploration Parameters
ε schedule 1.0→0.05 over 300000; ε at max_steps=2000 reported in exploration study.

## Performance Metrics
Avg steps/ep: 647.8; loss curve & reward MA plotted.

## Q-Learning Classification
Value-based: learn Q(s,a) and derive a greedy/ε-greedy policy.

## Q-Learning vs. LLM Agents
DQN uses dense step-wise rewards in an MDP; LLM agents use preference/sparse signals and plan via language/tools.

## Expected Lifetime Value
E[Σ γ^t r_t] from (s,a) (discounted future reward), estimated via Bellman backup.

## RL → LLM Agents
Value estimates and exploration ideas apply to tool-using LLM agents.

## Planning (RL vs LLM)
RL: model-based rollouts/value iteration; LLM: chain/tree-of-thought with tools.

## Algorithm
Q ← Q + α [r + γ max_a' Q_target(s',a') − Q(s,a)] ; Huber loss; soft target updates.

## LLM Integration
Planner–Controller (LLM sets subgoals; DQN executes) and World-model + Controller.
