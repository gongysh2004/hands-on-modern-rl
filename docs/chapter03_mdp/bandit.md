# 动手：两台老虎机——第一个 RL 决策

多臂老虎机问题（Multi-Armed Bandit, MAB）是 RL 最经典的教学起点。Thompson 在 1933 年最早提出了贝叶斯视角的解法 [^1]，Robbins 在 1952 年将其形式化为序贯决策问题 [^2]。它用一个极简的环境暴露了 RL 最核心的决策困境：**探索与利用（Exploration vs. Exploitation）**。

前两章的实践（CartPole 和 DPO）已经隐含了这个困境——PPO 在训练过程中需要不断尝试新策略（探索），同时利用已学到的知识获取高分（利用）。但在那些实验中，探索机制被封装在算法内部。本章让我们亲手面对这个困境。

## 两台老虎机

你面前有两台老虎机。每台投 1 枚硬币，中奖吐出 2 枚（净赚 +1），不中奖吞掉（净亏 -1）。你有 100 次机会。你不知道两台机器的出奖概率。

这个设定看似简单，但它精确地描述了 RL 的核心矛盾：**你需要"试试"来搞清楚哪个更好（探索），但你又想"锁定"更好的那个来赚更多（利用）。** 探索太多浪费机会，太少则可能永远锁定在次优选择上。

这个困境无处不在：选餐厅（老店还是新店）、看电影（熟悉的类型还是陌生的推荐）、甚至科研选题（深耕一个方向还是跨界尝试）。RL 把这个日常决策变成了可以被精确分析和优化的问题。

## 三种策略，三种结局

为了使讨论更具体，假设一个上帝视角的事实：A 台出奖率 60%，B 台只有 40%。但玩家不知道——只能通过尝试来发现。

### 策略 1：闭眼随机（50/50）

每次随机选一台。期望回报：

$$\mathbb{E}[R] = 0.5 \times 0.2 + 0.5 \times (-0.2) = 0$$

跑 100 轮，总回报在 0 附近波动。原因：一半时间拉了 A（期望每次赚 0.2），一半时间拉了 B（期望每次亏 0.2），互相抵消。

### 策略 2：永远拉 A（已知最优）

如果你知道 A 更好，永远拉 A：

$$\mathbb{E}[R] = 0.6 \times (+1) + 0.4 \times (-1) = 0.2$$

每轮平均赚 0.2，100 轮净赚约 20。但现实中你不知道——你需要通过尝试来发现。

### 策略 3：先试后定

前 20 轮两边交替试（探索），记录各自的胜率；后 80 轮锁定表现更好的那台（利用）。期望回报约 0.16——不如透视眼的 0.20（前 20 轮有探索成本），但比随机好得多。偶尔探索阶段运气差时会错误地锁定 B，这是探索不足的风险。

## 探索策略的对比

"先试后定"只是探索策略的一种。不同策略有不同的探索机制和代价：

| **策略** | **做法** | **探索机制** | **缺点** |
| --- | --- | --- | --- |
| 纯随机 | 均匀采样 | 无 | 永远学不到最优 |
| 贪心 | 永远选当前估计最优 | 无 | 可能锁定次优 |
| ε-贪婪 | 以 ε 概率随机，1-ε 选最优 | 固定比例探索 | ε 不随时间衰减 |
| 先试后定 | 前 N 步探索后利用 | 预算制 | N 不好选 |
| UCB | 选"估计均值 + 不确定性"最高的 | 不确定性驱动 | 需要维护置信区间 |
| Thompson Sampling | 从后验分布采样 | 概率匹配 | 需要贝叶斯更新 |

其中 UCB（Upper Confidence Bound [^3]）和 Thompson Sampling [^1] 是理论上最优的策略，它们的 Regret（遗憾值）随时间以对数速率增长，达到了 Lai & Robbins (1985) [^4] 证明的下界。

### Regret：衡量探索成本

Regret 形式化了"探索的代价"：

$$R_T = T \mu^* - \sum_{t=1}^T \mu_{a_t}$$

其中 $\mu^*$ 是最优臂的期望回报，$\mu_{a_t}$ 是第 $t$ 步所选臂的期望回报。Regret 衡量的是：与"一开始就知道最优"相比，你的策略累计损失了多少。好的探索策略应该让 $R_T$ 增长得尽可能慢。

## 用 Python 搭建老虎机

```python
import random

class TwoArmedBandit:
    """两台老虎机：最简 RL 环境"""

    def __init__(self, prob_a=0.6, prob_b=0.4):
        self.prob_a = prob_a
        self.prob_b = prob_b

    def pull(self, arm):
        """拉某一台机器，返回奖励"""
        if arm == "A":
            return 1 if random.random() < self.prob_a else -1
        else:
            return 1 if random.random() < self.prob_b else -1
```

这个环境没有"状态"——不管你上一步拉了哪台机器，这一步面对的情况一模一样。这就是老虎机的特点：它是一个**单状态 MDP**。后面我们会看到 CartPole 和 LLM 就不是这样了——它们的状态会随着动作而改变。

## 跑一把看看

```python
# ==========================================
# 策略 1：闭眼随机
# ==========================================
env = TwoArmedBandit()
total = sum(env.pull(random.choice(["A", "B"])) for _ in range(100))
print(f"随机策略 100 轮总回报: {total}，平均: {total/100:.2f}")

# ==========================================
# 策略 2：永远拉 A
# ==========================================
total = sum(env.pull("A") for _ in range(100))
print(f"透视眼策略 100 轮总回报: {total}，平均: {total/100:.2f}")

# ==========================================
# 策略 3：先试后定
# ==========================================
rewards_a, rewards_b = [], []
total = 0
for i in range(100):
    if i < 20:
        arm = "A" if i % 2 == 0 else "B"
    else:
        avg_a = sum(rewards_a) / len(rewards_a) if rewards_a else 0
        avg_b = sum(rewards_b) / len(rewards_b) if rewards_b else 0
        arm = "A" if avg_a >= avg_b else "B"

    reward = env.pull(arm)
    total += reward
    (rewards_a if arm == "A" else rewards_b).append(reward)

print(f"先试后定 100 轮总回报: {total}，平均: {total/100:.2f}")
```

预期输出：

```
随机策略 100 轮总回报: -2，平均: -0.02
透视眼策略 100 轮总回报: 18，平均: 0.18
先试后定 100 轮总回报: 14，平均: 0.14
```

三种策略，同一台机器，结果差异显著。**策略决定了你能从环境中拿走多少价值。**

## 期望回报：衡量策略的标尺

期望回报 $\mathbb{E}[R]$ 把"策略好不好"从一个模糊的感觉变成了一个精确的数字：

| **策略** | **计算** | **期望回报** |
| --- | --- | --- |
| 随机 50/50 | $0.5 \times 0.2 + 0.5 \times (-0.2)$ | 0 |
| 永远拉 A | $0.6 \times 1 + 0.4 \times (-1)$ | +0.2 |
| 永远拉 B | $0.4 \times 1 + 0.6 \times (-1)$ | -0.2 |

期望回报越高，策略越好。这个数字不是某一次的运气，而是大量实验的平均趋势——就像掷骰子的期望值是 3.5，你永远掷不出 3.5，但大量实验的平均会趋近它。

一个重要洞察：同样是"永远拉 A"，如果两台机器出奖率都是 50%，期望回报变成 0——和随机拉没区别。**策略的好坏取决于环境有没有可以被利用的结构。** 如果环境是公平的，再聪明的策略也没用；如果环境有偏（A 比 B 好），好策略才能体现优势。RL 的本质，就是发现并利用环境的结构。

## Agent-Environment 交互循环

不管你是拉老虎机、控制 CartPole、还是训练大模型，RL 的交互模式都遵循同一个循环：

```mermaid
flowchart LR
    Agent("Agent（智能体）") -->|"动作 a：'拉 A'"| Env("Environment（环境）")
    Env -->|"新状态 s'"| Agent
    Env -->|"奖励 r：+1（中了）"| Agent

    style Agent fill:#fff3e0,stroke:#f57c00,color:#000
    style Env fill:#e8f5e9,stroke:#388e3c,color:#000
```

智能体选择动作，环境给出奖励和新状态，循环往复。在老虎机中，动作是"拉 A 或拉 B"，奖励是 ±1。在 CartPole 中，动作是"左推或右推"，状态是 4 个物理量，奖励是每步 +1。在 DPO 对齐大模型时，动作是"下一个 token"，奖励是"人类偏好打分"。表面上千差万别，底层是同一个循环。

第 2 章做 DPO 时，模型就是那个智能体。它选 token（动作），被偏好信号（奖励）引导，最终学会了"说什么更受欢迎"。你其实已经做过 RL 了——只不过当时被封装在 TRL 库的黑盒里。

## 从直觉到数学

这个简单的老虎机游戏已经暴露了 RL 的两个核心问题：

**策略的好坏取决于环境。** 好的策略需要"看懂"环境结构，然后选择能最大化收益的行动。

**期望回报可以量化策略好坏。** $\mathbb{E}[R]$ 把"策略好不好"从一个模糊的感觉变成了一个精确的数字。这是后续所有 RL 理论的基石。

但现在的期望回报只考虑了单步——拉一次，得一个奖励。真实的 RL 问题往往涉及多步决策：CartPole 要活 200 步，大模型要生成 500 个 token。多步的情况下，"长期的总收益"怎么定义？眼前的 1 分和 10 步后的 1 分一样值钱吗？策略本身又该怎么形式化地描述？

下一节将把"状态、动作、奖励"这些直观概念提炼成精确的数学框架。[MDP 五元组、折扣回报与策略](./mdp)

## 参考文献

[^1]: Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. _Biometrika_, 25(3/4), 285-294.

[^2]: Robbins, H. (1952). Some aspects of the sequential design of experiments. _Bulletin of the American Mathematical Society_, 58(5), 527-535.

[^3]: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. _Machine Learning_, 47(2-3), 235-256.

[^4]: Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. _Advances in Applied Mathematics_, 6(1), 4-22.
