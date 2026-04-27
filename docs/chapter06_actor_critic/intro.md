# 第6章：Actor-Critic——两条路线的融合

第 4 章走了路线一（Value-Based）：学 $Q(s,a)$，选分数最高的。打分准确，但不擅长探索，且只能处理离散动作。第 5 章走了路线二（Policy-Based）：直接优化 $J(\theta)$。擅长探索、支持连续动作，但方差太大——同一策略跑两次，梯度估计可能天差地别。

上一章末尾我们发现了一个关键线索：减掉基线可以降低方差，而最好的基线就是 $V(s)$。但 $V(s)$ 本身也需要学习——需要一个专门的网络来估计它。这个网络就是 **Critic**。

本章将把两条路线拼在一起：用路线一的方法训练一个 Critic 来评估动作好坏，用路线二的方法训练一个 Actor 来选动作。这就是 **Actor-Critic 架构**。

::: tip 前置知识回顾
本章是前面所有章节的综合运用：

- [V(s) 与贝尔曼方程](../chapter03_mdp/value-v)——Critic 的理论基础
- [DP/MC/TD 速览](../chapter03_mdp/value-v)——训练 Critic 的三种方法
- [TD Error](../chapter03_mdp/value-v)——Critic 的训练信号
- [路线二：J(θ)](../chapter03_mdp/policy-objective)——Actor 的优化目标
- [基线实验](../chapter05_policy_gradient/baseline-experiment)——为什么需要 V(s) 作为基线
  :::

## 本章结构

| 小节                                 | 核心问题                                            |
| ------------------------------------ | --------------------------------------------------- |
| [优势函数](./advantage-function)     | 优势函数是什么？为什么它比 $G_t$ 更好？             |
| [Critic 训练方法](./critic-training) | 怎么训练 Critic 来估计 $V(s)$？DP/MC/TD 的具体实现  |
| [Actor-Critic 架构](./actor-critic)  | Actor 和 Critic 怎么协作？TD Error 如何替代 $G_t$？ |
| [项目：AlphaGo 简易复现](./alphago)  | Actor-Critic + MCTS 能做什么？                      |

让我们从优势函数开始——它是连接 Actor 和 Critic 的桥梁。[优势函数与 Critic 训练](./advantage-critic)
