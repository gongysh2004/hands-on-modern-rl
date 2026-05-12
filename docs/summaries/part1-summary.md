# Part 1: 实践基础 — 核心概念回顾

## 这一 Part 我们学了什么？

在前面两章中，我们从零运行了两个完整的强化学习训练实验。第 1 章用 Stable Baselines3 训练了一个 CartPole 倒立摆智能体——这是强化学习界的 "Hello World"。第 2 章用 TRL 库对 Qwen2.5-0.5B 大模型做了一次 DPO 偏好对齐，亲眼看到模型学会区分好回答和坏回答。

两章加起来，我们一共掌握了以下内容：

- **核心循环**：强化学习的本质是"观察状态 → 选动作 → 获得奖励 → 更新策略"的循环。无论多复杂的算法，都是在这个循环上做文章。
- **策略**：策略 $\pi(a|s)$ 就是"给定当前状态，选择每个动作的概率"。训练的目标就是让好动作的概率越来越高。
- **策略梯度**：用一个简洁的公式 $\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a|s) \cdot G$ 告诉我们——如果某个动作的回报 $G$ 是正的，就增加这个动作的概率。
- **DPO 损失**：$\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})$。核心思想是让好回答的隐式奖励高于坏回答，从而绕过奖励模型。
- **两个工具链**：传统 RL 用 Gymnasium + Stable Baselines3；大模型对齐用 Transformers + TRL。

下面让我们逐章复习这些内容。

## 第 1 章：CartPole——你的第一个 RL 程序

### Agent-Environment 交互循环

强化学习的本质是一个智能体（Agent）与环境（Environment）不断交互的过程。想象你在玩一个游戏：你看到屏幕上的画面（状态），按下按键（动作），游戏给你分数（奖励），然后画面更新。你不断重复这个过程，分数越来越高——这就是强化学习。

在 CartPole 任务中，智能体观察到的状态 $s$ 是一个 4 维向量：小车的位置和速度、杆子的角度和角速度。智能体只能在两个动作中选择——向左推或向右推，即 $\mathcal{A} = \{0, 1\}$。每存活一个时间步，智能体获得 +1 的奖励；杆子倒下，回合结束。

整个交互可以用一个简洁的循环来描述：

```python
obs, info = env.reset()
while True:
    action = model.predict(obs)        # 策略：根据状态选动作
    obs, reward, done, truncated, info = env.step(action)  # 环境返回新状态和奖励
    if done or truncated:
        break
```

这个"观察 → 决策 → 动作 → 奖励"的循环就是强化学习的核心范式。无论后面我们用多复杂的算法——DQN、PPO 还是 DPO——本质上都是在这个循环上做文章：如何让策略 $\pi(a|s)$ 越来越好？

### 策略与策略梯度

策略 $\pi_\theta(a|s)$ 是一个从状态到动作概率分布的映射。在 CartPole 中，我们的策略是一个两层 64 个神经元的全连接网络，输入 4 个状态值，输出 2 个动作的 logits，经过 softmax 转换为概率。

训练这个策略的方式是**策略梯度**。直觉非常简单：如果一个动作带来了好的结果（总回报 $G_t$ 大），就让这个动作的概率升高；反之就降低。

具体怎么做？我们的目标是最大化期望回报 $J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$。$\theta$ 是神经网络的参数，它唯一控制的就是策略 $\pi_\theta(a|s)$——每个动作的概率。对 $J$ 求 $\theta$ 的梯度，第一步得到 $\sum_a \nabla_\theta \pi_\theta(a|s) \cdot G_t$。但这里有个问题：$\nabla \pi$ 不是概率分布，你没法从里面采样——而实际训练只能靠采样。

解决方案是一个数学技巧：$\nabla \pi = \pi \cdot \nabla \log \pi$（链式法则）。把 $\pi$ 乘回去，梯度就变回了期望形式：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

现在可以采样了。跑一局游戏拿到一组样本 $(s_t, a_t, G_t)$，用**单次采样近似期望**：

$$\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

直觉上：$\nabla_\theta \log \pi_\theta(a_t|s_t)$ 是"调概率的旋钮"，$G_t$ 是"往哪个方向调"——回报为正就调高概率，为负就调低。$\log$ 还有两个实用的好处：把概率的乘法变成加法（数值更稳定），以及 $\nabla \log \pi = \nabla \pi / \pi$ 自然地对低概率动作赋予更大的梯度，鼓励探索。完整的推导在 [第 5 章](../chapter05_policy_gradient/reinforce)。

对应的损失函数就是 $\mathcal{L} = -\log \pi_\theta(a_t|s_t) \cdot G_t$。注意前面的负号——我们做的是梯度*下降*，所以要最小化这个损失来最大化期望回报。这就是 REINFORCE 算法的核心。

在训练过程中，你会观察到两个关键指标呈现"剪刀交叉"的现象：回合奖励（Reward）逐步上升，说明策略越来越强；策略熵（Entropy）$\mathcal{H} = -\sum_a \pi(a|s) \log \pi(a|s)$ 逐步下降，说明策略从随机探索变得越来越确定。这是 RL 训练健康的标志。

### 用 Stable Baselines3 训练

得益于现代开源生态，完成上面这一切只需要几行代码：

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 创建环境和智能体
env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)

# 训练——就这一行
model.learn(total_timesteps=80000)

# 评估效果
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
```

这里我们使用的 PPO 算法是目前最主流的策略梯度方法，它的核心思想会在第 6 章详细展开。现在只需要知道：PPO 在策略梯度基础上加了**裁剪机制**，防止策略更新步子太大导致训练崩溃。

## 第 2 章：DPO——让大模型学会"说好话"

### 大模型训练的三阶段

现代大模型的训练通常分三个阶段。第一阶段是**预训练**（Pre-training），模型在海量文本上学习"预测下一个词"，相当于"博览群书"。第二阶段是**监督微调**（SFT），用人工编写的高质量对话数据教模型遵循指令，相当于"学习礼仪"。第三阶段是**强化学习对齐**（RL Alignment），让模型学会区分好回答和坏回答，相当于"树立三观"。

DPO（Direct Preference Optimization）就是第三阶段的一种核心方法。

### DPO 损失函数

DPO 的核心思想非常优雅：与其先训练一个奖励模型再用 RL 优化（传统的 RLHF），不如直接用偏好数据来优化策略。

假设我们有一组偏好数据 $(x, y_w, y_l)$，其中 $x$ 是提示词，$y_w$ 是人类认为好的回答，$y_l$ 是差的回答。DPO 定义了一个**隐式奖励**：

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

其中 $\pi_\theta$ 是正在训练的模型，$\pi_{\text{ref}}$ 是训练前的参考模型（冻结不动）。$\beta$ 是一个超参数，控制模型允许偏离参考模型多远——$\beta$ 越大，模型越保守。

什么是"隐式奖励"？传统 RLHF 需要单独训练一个奖励模型来给回答打分。DPO 发现，这个分数其实已经藏在策略模型的概率比值里了——如果模型现在比训练前更喜欢某个回答（$\pi_\theta > \pi_{\text{ref}}$），隐式奖励就是正的；如果更喜欢了（$\pi_\theta < \pi_{\text{ref}}$），就是负的。

有了隐式奖励，DPO 的损失函数就是让好回答的奖励高于坏回答的奖励：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

其中 $\sigma$ 是 sigmoid 函数。直觉上，这就像一个分类器：给定一对回答，模型要正确判断哪个更好。当好回答的隐式奖励远高于坏回答时，$\sigma$ 内部的值很大，$\log \sigma$ 接近 0，损失很小。反过来，如果模型还分不清好坏，损失就大，梯度就会推动模型朝正确方向更新。

### 用 TRL 训练 DPO

实际训练中，我们使用 HuggingFace 的 TRL 库，整个过程同样非常简洁：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# 加载模型——一个是训练的，一个是冻结的参考
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 配置训练参数
training_args = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=3,
    beta=0.1,       # KL 惩罚系数
)

# 开始训练
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=preference_dataset,  # 包含 prompt, chosen, rejected 三列
    processing_class=tokenizer,
)
trainer.train()
```

训练过程中需要关注两个关键指标：**训练损失**应该逐步下降，说明模型在学习区分好坏回答；**奖励边界**（Reward Margin，即好回答与坏回答的隐式奖励之差）应该逐步增大，说明模型的区分能力在增强。

## 小结

经过这两章的学习，我们建立了两个重要的认知。第一，强化学习的核心是 Agent-Environment 交互循环——策略根据状态选动作，环境返回奖励和新状态，策略根据奖励信号自我改进。第二，RL 既可以用于传统控制任务（CartPole），也可以用于大模型对齐（DPO），两者共享同一套底层逻辑：定义一个目标函数，用梯度下降来优化。

在接下来的 Part 2 中，我们将深入这套逻辑的数学基础——MDP、贝尔曼方程、DQN、策略梯度定理、PPO——为后面的 LLM 对齐打下坚实的理论基础。

> **下一站**：[Part 2: 理论与方法](/chapter03_mdp/intro)
