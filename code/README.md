# 代码索引

本目录包含课程各章的配套代码。每章代码可独立运行。

## 快速开始

```bash
# 全局安装（推荐先创建虚拟环境）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装某一章的依赖
pip install -r chapter01_cartpole/requirements.txt
```

## 章节代码一览

### Part 1: 极速入门

| 章节                    | 目录                  | 代码文件                  | 说明                                             |
| ----------------------- | --------------------- | ------------------------- | ------------------------------------------------ |
| **Ch01** 传统 RL 初体验 | `chapter01_cartpole/` | `hello_rl.py`             | SB3 版 CartPole 训练与演示                       |
|                         |                       | `hello_rl_tensorboard.py` | 带 TensorBoard 日志的训练版本                    |
|                         |                       | `pytorch_from_scratch.py` | 纯 PyTorch REINFORCE 实现（黑盒拆解）            |
|                         |                       | `requirements.txt`        | gymnasium, stable-baselines3, torch, tensorboard |
| **Ch02** 现代 RL 初体验 | `chapter02_dpo/`      | `0-generate_data.py`      | 偏好数据生成脚本                                 |
|                         |                       | `1-test_before.py`        | 微调前测试                                       |
|                         |                       | `2-train_dpo.py`          | DPO 训练                                         |
|                         |                       | `3-test_after.py`         | 微调后测试                                       |

### Part 2: 理论与方法

| 章节                             | 目录                         | 代码文件   | 说明                                                       |
| -------------------------------- | ---------------------------- | ---------- | ---------------------------------------------------------- |
| **Ch03** MDP 与大模型语境        | `chapter03_mdp/`             | _(待补充)_ | 猜硬币 MDP 游戏、GridWorld Q-Learning、贝尔曼方程验证      |
| **Ch04** 深度强化学习 DQN        | `chapter04_dqn/`             | _(待补充)_ | DQN 玩 Pong、经验回放与目标网络                            |
| **Ch05** 策略梯度与 Actor-Critic | `chapter05_policy_gradient/` | _(待补充)_ | REINFORCE + 基线对比 + Actor-Critic、离散/连续动作空间示例 |
| **Ch06** PPO 与奖励模型          | `chapter06_ppo/`             | _(待补充)_ | PPO 完整 PyTorch 实现（LunarLander）、GAE 调参、RM 训练    |

### Part 3: LLM 时代

| 章节                                     | 目录                   | 代码文件   | 说明                                                   |
| ---------------------------------------- | ---------------------- | ---------- | ------------------------------------------------------ |
| **Ch07** 对齐方法族（DPO / KTO / SimPO） | `chapter07_alignment/` | _(待补充)_ | DPO 数学推导与训练、KTO 标签训练、SimPO 无参考模型训练 |
| **Ch08** GRPO、DAPO 与 RLVR              | `chapter08_grpo_rlvr/` | _(待补充)_ | GRPO + GSM8K 数学推理、DAPO 动态采样、RLVR 验证奖励    |

### Part 4: 进阶与前沿

| 章节                     | 目录                            | 代码文件   | 说明                                          |
| ------------------------ | ------------------------------- | ---------- | --------------------------------------------- |
| **Ch09** 连续动作控制    | `chapter09_continuous_control/` | _(待补充)_ | PPO/TD3 + HalfCheetah、高斯策略 vs 确定性策略 |
| **Ch10** RLHF 完整流水线 | `chapter10_rlhf/`               | _(待补充)_ | RLHF 三阶段流水线 + RLAIF                     |
| **Ch11** VLM 强化学习    | `chapter11_vlm_rl/`             | _(待补充)_ | GRPO 训练 VLM                                 |
| **Ch12** Agentic RL      | `chapter12_agentic_rl/`         | _(待补充)_ | 工具调用智能体、多轮交互 RL                   |
| **Ch13** 未来趋势        | `chapter13_future_trends/`      | _(无代码)_ | 纯理论章节                                    |

### 附录

| 附录                       | 目录                        | 代码文件   | 说明                                |
| -------------------------- | --------------------------- | ---------- | ----------------------------------- |
| **附录A** 常见坑与解法     | `appendix_common_pitfalls/` | _(待补充)_ | 常见故障复现脚本                    |
| **附录B** 工业级训练与评测 | _(待补充)_                  | _(待补充)_ | 集群训练与评测                      |
| **附录C** 算法速查         | `appendix_algorithm_guide/` | _(无代码)_ | 算法选型指南 + 现代 RL 框架三层架构 |
