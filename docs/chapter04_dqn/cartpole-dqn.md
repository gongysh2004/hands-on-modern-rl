# 4.3 动手：用 DQN 玩 CartPole

前面两节我们理清了 DQN 的理论框架和三个核心组件。现在让我们把它变成代码。我们选择 CartPole 作为实验环境——和第 1 章不同的是，当时我们用 Stable Baselines3 的黑盒 `PPO("MlpPolicy", env)` 一行完成训练，无需了解内部实现。现在，我们要用从第 3 章一路学来的知识，亲手搭建 DQN 的每一个组件。

为什么不用 Atari？Atari 需要图像预处理（裁剪、灰度化、帧堆叠）和 CNN 网络，这些额外的工程细节会分散注意力。CartPole 的输入是 4 维向量，一个简单的 MLP 就能处理，让我们把精力集中在 DQN 算法本身。等理解了 CartPole 上的 DQN，迁移到 Atari 只需要换网络结构和预处理流程。

## CartPole 长什么样？

在动手写代码之前，让我们先看看 CartPole 的问题到底长什么样。一根杆子铰接在小车上，杆子初始时接近直立。你可以控制小车向左或向右施加推力。目标很简单：让杆子尽可能久地保持直立不倒。

```
训练前（Episode 1）               训练后（Episode 300）

    |  ← 杆子立刻倒下                |||||||  ← 杆子稳稳直立
    |                               |||||||
   /                               |||||||
  ┌───┐                            ┌───┐
  │   │  ← 小车乱动                │   │ ← 小车精准微调
──┴───┴──                        ──┴───┴──
─────────────                    ────────────

奖励：9.4 步就倒下                 奖励：500 步（满分）
```

CartPole 的状态有 4 个维度，你不需要理解物理细节，只需要知道它们描述了"杆子此刻有多歪、小车在哪里"：

| 状态分量   | 符号           | 含义                   | 直觉                       |
| ---------- | -------------- | ---------------------- | -------------------------- |
| 小车位置   | $x$            | 小车在轨道上的水平位置 | "车在轨道中间还是偏了"     |
| 小车速度   | $\dot{x}$      | 小车的水平移动速度     | "车在往哪边溜"             |
| 杆子角度   | $\theta$       | 杆子偏离竖直方向的角度 | "杆子歪了多少"             |
| 杆子角速度 | $\dot{\theta}$ | 角度的变化速率         | "杆子在往哪边倒、倒得多快" |

每个时间步，智能体只能做一个选择——向左推或者向右推。杆子保持直立每步得 +1 分，杆子倒下（角度超过 $\pm 12°$）或者小车滑出屏幕就游戏结束。满分 500——意味着杆子稳稳立了 500 步。

## 完整代码：从零实现 DQN

下面是完整的 DQN 实现，大概 150 行代码。我们会分几段来写，每段配有详细解读。

### 第一部分：Q-Network 和经验回放

```python
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. Q-Network：输入状态，输出每个动作的 Q 值
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 经验回放池
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)
```

Q-Network 是一个简单的三层 MLP：4 维输入 → 128 隐藏 → 128 隐藏 → 2 维输出。经验回放池用 `deque` 实现，容量 10000 条——超过容量后旧经验自动淘汰。

### 第二部分：DQN 智能体

```python
# ==========================================
# 3. DQN 智能体
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500,
                 buffer_capacity=10000, batch_size=64, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # ε-贪婪策略：ε 从 1.0 线性衰减到 0.01
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Q-Network 和目标网络
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())  # 初始参数一致
        self.target_net.eval()  # 目标网络不参与训练

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 经验回放池
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state):
        """ε-贪婪策略选择动作"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)  # 随机探索
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_net(state_tensor)
                return q_values.argmax().item()  # 选 Q 值最大的动作

    def update(self):
        """从经验回放池采样并更新 Q-Network"""
        if len(self.buffer) < self.batch_size:
            return 0.0  # 经验不够，不更新

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # 计算 Q(s, a)：网络对当前状态的输出，只取选定动作的 Q 值
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算 TD Target：r + γ max Q(s', a'; θ⁻)
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(dim=1)[0]
            td_target = rewards + self.gamma * next_q_max * (1 - dones)

        # 计算 Loss 并更新
        loss = self.loss_fn(q_values, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """将 Q-Network 的参数复制到目标网络"""
        self.target_net.load_state_dict(self.q_net.state_dict())
```

这段代码里有几个值得注意的细节。

`select_action` 使用了 $\varepsilon$-贪婪策略，其中 $\varepsilon$ 按指数衰减——训练初期 $\varepsilon \approx 1$，几乎纯随机探索；训练后期 $\varepsilon \approx 0.01$，几乎完全利用学到的知识。这和第 3 章 GridWorld 中用的 $\varepsilon$-贪婪策略完全一样，只是衰减方式不同。

`update` 中的 `.gather(1, actions.unsqueeze(1))` 是 PyTorch 中的高级索引操作——从网络输出的所有动作 Q 值中，只取出实际执行的那个动作的 Q 值。`(1 - dones)` 这一项处理了 episode 结束的情况：如果 `done=True`，意味着没有"下一状态"，TD Target 就等于即时奖励 $r$。

### 第三部分：训练循环

```python
# ==========================================
# 4. 训练循环
# ==========================================
env = gym.make("CartPole-v1")
agent = DQNAgent(state_dim=4, action_dim=2)

num_episodes = 300
reward_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        # 选择并执行动作
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # 存入经验回放池
        agent.buffer.push(state, action, reward, next_state, float(done))

        # 更新 Q-Network
        agent.update()

        # 每隔 target_update 步更新目标网络
        if agent.steps_done % agent.target_update == 0:
            agent.update_target()

        state = next_state
        if done:
            break

    reward_history.append(total_reward)

    # 打印训练进度
    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"Episode {episode+1}/{num_episodes} | "
              f"最近50轮平均奖励: {avg:.1f} | "
              f"ε: {agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(-agent.steps_done / agent.epsilon_decay):.3f}")

env.close()
```

训练循环的逻辑很直白：每一步先选动作、执行动作、存经验、更新网络。每隔固定步数同步目标网络。每个 episode 结束后记录总奖励。

### 第四部分：测试训练好的智能体

```python
# ==========================================
# 5. 测试：用训练好的 DQN 玩 CartPole
# ==========================================
test_env = gym.make("CartPole-v1")
state, _ = test_env.reset()
total_reward = 0

while True:
    # 训练完成后不再探索，纯利用
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = agent.q_net(state_tensor).argmax().item()
    state, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

test_env.close()
print(f"\n测试得分: {total_reward}")
```

测试时我们关闭了探索——不再用 $\varepsilon$-贪婪，而是直接选 Q 值最大的动作。如果训练成功，测试得分应该接近 CartPole 的满分 500。

## 预期输出

运行完整代码后，你会看到类似这样的训练日志：

```
Episode 50/300 | 最近50轮平均奖励: 22.5 | ε: 0.741
Episode 100/300 | 最近50轮平均奖励: 85.3 | ε: 0.301
Episode 150/300 | 最近50轮平均奖励: 182.7 | ε: 0.089
Episode 200/300 | 最近50轮平均奖励: 312.4 | ε: 0.023
Episode 250/300 | 最近50轮平均奖励: 415.8 | ε: 0.011
Episode 300/300 | 最近50轮平均奖励: 465.2 | ε: 0.010

测试得分: 500.0
```

训练过程展现出典型的 DQN 学习曲线：前 50 轮平均奖励很低（~22），智能体几乎无法保持平衡。然后随着探索逐渐减少、经验回放池中积累的经验越来越多，性能开始稳步上升。200 轮左右突破 300 分，300 轮时接近满分。最终测试得分为 500——CartPole 的最高分，意味着杆子在 500 步内完全没有倒下。

## 可视化：画出训练曲线

训练日志里的数字太抽象了，让我们把 reward 曲线画出来。在训练循环结束后加入以下代码：

```python
import matplotlib.pyplot as plt

# ==========================================
# 6. 绘制训练曲线
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：每轮总奖励（灰色散点 + 滑动平均红线）
axes[0].plot(reward_history, color='lightgray', alpha=0.6, label='每轮奖励')
window = 20
if len(reward_history) >= window:
    moving_avg = [np.mean(reward_history[max(0, i-window):i+1])
                  for i in range(len(reward_history))]
    axes[0].plot(moving_avg, color='red', linewidth=2, label=f'{window}轮滑动平均')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward')
axes[0].set_title('DQN 训练曲线（CartPole）')
axes[0].legend()
axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.5, label='满分 500')

# 右图：ε 衰减曲线
epsilons = [agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) *
            np.exp(-s / agent.epsilon_decay) for s in range(agent.steps_done)]
axes[1].plot(epsilons, color='blue')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('ε (探索率)')
axes[1].set_title('ε-贪婪策略的衰减过程')
axes[1].axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, label='ε 下限 0.01')
axes[1].legend()

plt.tight_layout()
plt.savefig('dqn_cartpole_training.png', dpi=150)
plt.show()
```

你会看到这样一张图：

```
┌─────────────────────────────┐  ┌──────────────────────────┐
│  DQN 训练曲线（CartPole）    │  │  ε-贪婪策略的衰减过程     │
│                             │  │ 1.0 ┤╲                   │
│ 500 ┤ ····满分线····         │  │     │  ╲                  │
│     │              ╱━━━━━   │  │     │   ╲                 │
│ 400 ┤           ╱━━         │  │     │    ╲                │
│     │       ╱━━━            │  │     │     ╲___            │
│ 200 ┤   ╱━━                 │  │     │         ╲_____      │
│     │  ╱                    │  │ 0.01┤──────────────━━━━━━  │
│  50 ┤╱····（散点很多波动）   │  │     └──────────────────── │
│     └────────────────────── │  │      Step                  │
│      Episode                │  │                            │
└─────────────────────────────┘  └──────────────────────────┘
```

左图展示了一个典型的 DQN 学习过程：前 50 个 episode 的灰色散点在 10-30 分之间乱窜，智能体几乎无法保持平衡。然后红色滑动平均线开始爬升——这是经验回放池积累到足够数据、Q 值开始收敛的标志。约 200 episode 后曲线加速上升，最终趋近满分 500。

右图展示了 $\varepsilon$ 的衰减过程：从一开始的 $\varepsilon \approx 1$（几乎纯随机探索），快速下降到 $\varepsilon \approx 0.01$（几乎纯利用）。注意 $\varepsilon$ 下降的速度很快——这意味着智能体在训练初期疯狂试错，收集各种经验填充回放池，然后迅速转向利用已学到的知识。这个"先探索后利用"的节奏，和人类学习新技能的过程很像。

你也可以把训练好的智能体录制成 GIF 动画，直观地看到它从"乱推"到"精准微调"的进步：

```python
# ==========================================
# 7. 录制训练前 vs 训练后的对比 GIF
# ==========================================
from gymnasium.utils.save_recording import save_as_gym_recording

# 用训练好的智能体跑一局
vis_env = gym.make("CartPole-v1", render_mode="rgb_array")
frames = []
state, _ = vis_env.reset()

for _ in range(500):
    frames.append(vis_env.render())
    with torch.no_grad():
        action = agent.q_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
    state, _, terminated, truncated, _ = vis_env.step(action)
    if terminated or truncated:
        break
vis_env.close()

# 保存为 GIF（需要 pip install imageio）
import imageio
imageio.mimsave('cartpole_trained.gif', frames, fps=30)
print(f"已保存 {len(frames)} 帧到 cartpole_trained.gif")
```

这段代码会生成一个 `cartpole_trained.gif` 动画，你可以看到训练后的智能体如何通过持续的微小推力调整，让杆子始终保持在直立状态。

这个学习过程和第 1 章用 SB3 的 PPO 看到的现象本质上是一样的——只是现在你能看到每一个零件在做什么。经验回放池里的每一条经验长什么样？目标网络多久更新一次？Q 值是怎么从随机噪声变成有意义的评估？这些在第 1 章都是黑盒，现在全部透明。

现在你已经跑通了一个完整的 DQN。接下来让我们深入观察训练过程中发生了什么——[训练日志分析](./training-analysis)。
