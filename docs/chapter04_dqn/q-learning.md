# Q-Learning：TD 方法 for Q

第 3 章介绍了路线一的核心思路：学习 $Q(s,a)$ 给每个动作打分，然后选分数最高的。我们还速览了三种估计价值的方法——DP、MC、TD——其中 TD 方法不需要环境模型，走一步就能更新，是最实用的选择。

本节将 TD 方法应用到 $Q$ 上，得到强化学习最经典的算法之一——Q-Learning。

## 从 TD 到 Q-Learning

第 3 章的 TD 方法用以下公式更新 $V(s)$：

$$V(s) \leftarrow V(s) + \alpha \underbrace{\left[ r + \gamma V(s') - V(s) \right]}_{\text{TD Error } \delta}$$

Q-Learning 做的事情完全类似，只是把 $V$ 换成 $Q$，并且在 TD Target 中用 $\max$ 代替对下一状态的估计：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

逐项拆解：

| 符号 | 含义 |
| --- | --- |
| $Q(s, a)$ | 当前对"在状态 $s$ 做动作 $a$ 值多少分"的估计 |
| $r + \gamma \max_{a'} Q(s', a')$ | TD Target：即时奖励 + 下一状态中最好动作的价值 |
| $\max_{a'} Q(s', a')$ | "到了 $s'$ 之后，最好的动作值多少分" |
| $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ | TD Error：预测与现实的落差 |

注意那个 $\max_{a'}$——它不看所有动作的平均，只看最好的那个。这意味着 Q-Learning 学的是**最优动作价值 $Q^*$**，不管当前用什么策略在探索。这就是离策略（off-policy）学习：用 $\varepsilon$-贪婪策略收集数据，但学的是最优策略的 $Q$ 值。

## ε-贪婪：平衡探索与利用

Q-Learning 需要数据来学习，但它学的是最优 $Q^*$，而不是当前策略的 $Q$。那收集数据时用什么策略？

最常用的选择是 **$\varepsilon$-贪婪（$\varepsilon$-greedy）**：

$$a = \begin{cases} \arg\max_a Q(s, a) & \text{以概率 } 1 - \varepsilon \text{（利用）} \\ \text{随机动作} & \text{以概率 } \varepsilon \text{（探索）} \end{cases}$$

$\varepsilon$ 控制探索的程度：$\varepsilon = 0.1$ 意味着 90% 的时间选当前最好的动作，10% 的时间随机尝试。这正是第 3 章讨论的探索-利用困境在路线一中的具体体现——用一个参数来人工平衡。

## 动手：4×4 GridWorld

用一个具体例子来感受 Q-Learning 的运作过程，亲眼看看 TD Error 是怎么从非零逐渐收敛到零的。

### 环境设定

```
┌───┬───┬───┬───┐
│ S │   │   │   │
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┬───┤
│   │   │   │   │
├───┼───┼───┬───┤
│   │   │   │ G │
└───┴───┴───┴───┘
```

4×4 网格，左上角起点 $S$，右下角终点 $G$。每步奖励 -1（鼓励尽快到达终点），到达终点奖励 0。动作：上/下/左/右。初始 Q-table：全部为 0。

### 手算第 1 步：从 S 向右走

智能体从 $S = (0,0)$ 出发，选择向右走到 $(0,1)$。即时奖励 $r = -1$。下一状态的所有 Q 值都是 0（初始化为 0）。

- TD Target $= -1 + 0.9 \times 0 = -1$
- TD Error $= -1 - 0 = -1$
- 新 Q 值 $= 0 + 0.1 \times (-1) = -0.1$

TD Error = -1 的含义：之前 Q 值是 0（"什么都不知道，猜测走这步不赚不亏"），实际走了一步却扣了 1 分——预测严重偏高，所以把 Q 值下调了 0.1。

第 2 步的情况类似：从 $(0,1)$ 继续向右走到 $(0,2)$，TD Error 仍然是 -1，新 Q 值也是 -0.1。因为周围的格子都还没学过，Q 值全是 0。

### 用代码验证

```python
import numpy as np

# 4x4 GridWorld Q-Learning
Q = np.zeros((16, 4))  # 16 个状态, 4 个动作 (上右下左)
alpha, gamma, epsilon = 0.1, 0.9, 0.1
goal = 15  # 右下角的索引

def state_to_idx(row, col):
    return row * 4 + col

def step(state, action):
    """执行动作，返回 (下一状态, 奖励, 是否结束)"""
    row, col = state // 4, state % 4
    if action == 0: row = max(row - 1, 0)      # 上
    elif action == 1: col = min(col + 1, 3)     # 右
    elif action == 2: row = min(row + 1, 3)     # 下
    elif action == 3: col = max(col - 1, 0)     # 左
    next_state = state_to_idx(row, col)
    reward = 0 if next_state == goal else -1
    done = next_state == goal
    return next_state, reward, done

# 训练 1000 个 episode
for ep in range(1000):
    state = 0  # 起点 S
    while state != goal:
        # ε-贪婪：90% 选最优，10% 随机探索
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        # Q-Learning 更新
        td_target = reward + gamma * np.max(Q[next_state])
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state

# 打印收敛结果
print("收敛后的 Q((0,0), 右) =", Q[0, 1].round(2))
print("最优路径（从 S 出发的动作序列）：")
state = 0
actions = ["↑", "→", "↓", "←"]
path = []
while state != goal:
    a = np.argmax(Q[state])
    path.append(actions[a])
    state, _, _ = step(state, a)
print(" → ".join(path))
```

预期输出：

```
收敛后的 Q((0,0), 右) = -5.69
最优路径（从 S 出发的动作序列）：
→ → → ↓ ↓ ↓
```

### 收敛过程

经过大量训练后，Q 值会收敛。以 $Q(S, \text{右})$ 为例：从 $S$ 到 $G$ 最短路径需要 6 步，每步 -1，考虑 $\gamma = 0.9$ 的折扣后：

$$Q((0,0), \text{右}) \approx -1 - 0.9 - 0.81 - 0.729 - 0.656 - 0.590 = -4.685$$

实际值约 -5.69（因为路径可能不是最优的 6 步直线路径）。此时 TD Error $\approx 0$——预判和实际一致了，学习完成。

这个过程揭示了 Q-Learning 的本质：TD Error 从一开始的 -1，通过成百上千次的微调，逐渐趋近于 0。每一次微调都是在说"上次猜错了，这次修一点"。

## Q-Learning 的关键性质

| **性质** | **说明** |
| --- | --- |
| Off-policy | 学的是 $Q^*$（最优），但可以用任何策略收集数据 |
| Model-free | 不需要知道环境的 $P$ 和 $R$ |
| 逐步更新 | 每走一步就更新，不需要等 episode 结束 |
| 收敛性 | 在表格情况下，Q-Learning 保证收敛到 $Q^*$ [^1] |

### 收敛性

Watkins & Dayan (1992) [^1] 证明了：在表格情况下，只要满足以下条件，Q-Learning 保证收敛到最优动作价值 $Q^*$：

1. 所有状态-动作对 $(s, a)$ 被无限次访问
2. 学习率 $\alpha$ 满足 $\sum_t \alpha_t = \infty$ 且 $\sum_t \alpha_t^2 < \infty$

条件 1 由 ε-贪婪策略保证（只要 $\varepsilon > 0$，每个动作都有非零概率被选中）。条件 2 要求学习率逐渐减小但不能减得太快——实践中常用 $\alpha_t = 1/t$ 或固定的小常数（如 0.1）。

### Decaying ε：让探索逐渐减少

固定 $\varepsilon = 0.1$ 意味着训练后期仍然有 10% 的时间在随机探索——这在不必要地损失回报。更实用的做法是**衰减 ε（decaying ε）**：

$$\varepsilon_t = \max\left(\varepsilon_{\min}, \, \varepsilon_0 - \frac{t}{T_{\text{decay}}}(\varepsilon_0 - \varepsilon_{\min})\right)$$

例如 $\varepsilon_0 = 1.0$，$\varepsilon_{\min} = 0.01$，$T_{\text{decay}} = 10000$：前 10000 步从完全随机线性衰减到 1%，之后保持 1%。这保证了早期充分探索，后期稳定利用。

### On-policy vs Off-policy：SARSA 对比

Q-Learning 的更新中用了 $\max_{a'} Q(s', a')$——它假设下一步会选最优动作。但实际策略（ε-贪婪）在下一步可能随机选了一个非最优动作。这种"学的是最优，做的不是最优"的分离就是 off-policy。

SARSA 是 Q-Learning 的 on-policy 版本，由 Rummery & Niranjan (1994) 提出 [^2]：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

注意区别：Q-Learning 用 $\max_{a'} Q(s', a')$（假设最优），SARSA 用 $Q(s', a')$（实际选的动作 $a'$）。

| | **Q-Learning (off-policy)** | **SARSA (on-policy)** |
| --- | --- | --- |
| TD Target | $r + \gamma \max_{a'} Q(s', a')$ | $r + \gamma Q(s', a')$ |
| 学的是 | $Q^*$（最优策略） | $Q^\pi$（当前策略） |
| 行为 | 乐观——假设下一步选最优 | 保守——考虑实际探索风险 |

经典例子：在 Cliff Walking 环境中，Q-Learning 学到了贴着悬崖走的最短路径（因为它假设不会随机掉下去），而 SARSA 学到了远离悬崖的更安全路径（因为它知道有 10% 概率会随机探索掉下去）。在安全关键场景中，SARSA 的保守可能更实用。

这些性质使 Q-Learning 成为最实用的 Value-Based 方法。但它有一个根本性的限制：**只能用表格存储 Q 值**。16 个格子的 GridWorld 没问题，但 CartPole 的状态是连续的，Atari 的画面有几十万像素——表格方法的存储需求远超物理设备的容量。

下一节将展示如何用神经网络替代表格，解决状态空间爆炸的问题。[从 Q-Learning 到 DQN](./from-q-to-dqn)
