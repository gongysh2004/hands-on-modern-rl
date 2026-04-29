# Ch3-7 细化计划（颗粒度版）

> 每一节列出具体要写的内容条目，颗粒度到"段落/公式/表格/引用"级别。
> 目标：每节从当前的 1-2 页扩充到 3-5 页，具备 Ch12 的纵深。

---

## Ch3：RL 通识——理论地图与全局视野

### 3.0 intro.md（章节导览）

**当前：** ~300 字，导航表
**目标：** ~800 字

具体内容条目：

1. 开篇段落：从第1-2章的实践出发，提出"PPO为什么能学会"这一核心问题（2段）
2. RL 的形式化定义：agent 在环境中通过交互最大化累积奖励的循环（1段 + mermaid 图）
3. 本章目标列表：读者学完后应能回答的 4-5 个问题（编号列表）
4. 章节导航表（列名加粗）
5. 不用"让我们从……开始"，用"下一节将……"

---

### 3.1 bandit.md（两台老虎机）

**当前：** ~2000 字，三种策略 + 代码
**目标：** ~3000 字

具体增加条目：

1. **多臂老虎机问题的历史**：Thompson (1933) 最早提出，Robbins (1952) 形式化（1段）
2. **探索策略的对比**：当前只有"先试后定"，增加一张表对比 4 种策略：

   | 策略     | 做法              | 探索机制 | 缺点           |
   | -------- | ----------------- | -------- | -------------- |
   | 纯随机   | 均匀采样          | 无       | 永远学不到     |
   | 贪心     | 永远选当前最优    | 无       | 可能锁定次优   |
   | ε-贪婪   | 以 ε 概率随机     | 固定比例 | ε 不随时间衰减 |
   | 先试后定 | 前 N 步探索后利用 | 预算制   | N 不好选       |

3. **Regret（遗憾值）的概念**：衡量探索成本的形式化工具，定义 $R_T = T \mu^* - \sum_{t=1}^T \mu_{a_t}$，一句话解释为什么它比期望回报更适合衡量学习效率
4. **参考文献**：Robbins (1952)、Sutton & Barto Chapter 2、Lai & Robbins (1985) asymptotic lower bound

保留：三种策略对比代码、Agent-Environment 循环、期望回报表

---

### 3.2 mdp.md（MDP 五元组 + G_t + 策略 π）

**当前：** ~2500 字
**目标：** ~3500 字

具体增加条目：

1. **MDP 的元组记法**：$\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$（对标 Ch12 的 $\langle S, A, P, R, \gamma \rangle$）
2. **马尔可夫性的深入讨论**：
   - 正式定义：$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)$
   - "历史无关性"的含义：为什么下象棋只需看当前局面
   - **反例**：什么情况下马尔可夫性不成立？（部分可观测、隐藏状态）
   - POMDP 的一句话预告（"后续章节的 Agentic RL 就是 POMDP"——回扣 Ch12）
3. **G_t 的数学性质**：
   - 当 $\gamma < 1$ 且 $|R| \leq R_{\max}$ 时，$G_t \leq \frac{R_{\max}}{1-\gamma}$（收敛性）
   - γ 的直觉解释：γ=0 是近视，γ→1 是远视，两者的 trade-off
4. **策略的正式讨论**：
   - 确定性策略 π: S → A 的映射
   - 随机性策略 π(a|s) 作为条件概率分布
   - 为什么 RL 中偏好随机性策略（探索 + 连续动作空间）
5. **参考文献**：Bellman (1957)、Puterman (1994) MDP theory

---

### 3.3 value-bellman.md（V(s) + 贝尔曼方程）

**从 value-v.md 拆出前半。**

**目标：** ~3000 字

具体内容条目：

1. **V(s) 的正式定义**：$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]$，下标 π 的含义
2. **直觉类比**：棋局评估器（保留），但改为陈述性语气
3. **贝尔曼方程的推导思路**（不是完整证明，而是思路）：
   - 从 V 的定义出发：$V^\pi(s) = \mathbb{E}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... | s_t = s]$
   - 拆出第一步：$= \mathbb{E}[r_t + \gamma (r_{t+1} + \gamma r_{t+2} + ...) | s_t = s]$
   - 认出括号内是 $G_{t+1}$：$= \mathbb{E}[r_t + \gamma G_{t+1} | s_t = s]$
   - 取条件期望：$= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a) + \gamma V^\pi(s')]$
4. **宝藏地图手算**（保留，改语气）
5. **老虎机 V=2.0 验证**（保留）
6. **贝尔曼最优方程**（当前缺失）：
   - 把 Σ 换成 max：$V^*(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')]$
   - 与期望方程的对比表
   - 为什么最优方程里没有 π
7. **参考文献**：Bellman (1957)

---

### 3.4 dp-mc-td.md（经典方法速览）

**从 value-v.md 拆出后半。**

**目标：** ~2500 字（每方法 2-3 段 + 1 个小例子）

每种方法按统一模板：

```
1. 核心思想（1段）
2. 更新公式（1个编号公式）
3. 一个具体例子（2-3句话的微型例子，不需要完整代码）
4. 局限（1段）
```

具体增加条目：

1. **DP**：
   - 策略评估的迭代：对所有状态反复应用贝尔曼方程
   - 例子：GridWorld 的 V 值在 3 次迭代后的变化（表格，不需要代码）
   - 局限：需要完整模型 + 状态空间爆炸
2. **MC**：
   - 首次访问 vs 每次访问的区别（一句话）
   - 例子：老虎机跑 100 次 episode 的 V 估计波动图（概念描述）
   - 局限：必须等 episode 结束 + 方差大
3. **TD(0)**：
   - MC 和 DP 的折中：采样（像 MC）+ 自举（像 DP）
   - 例子：随机游走（Random Walk）中 TD 和 MC 的学习曲线对比（概念描述）
   - 优势：逐步更新 + 低方差
4. **TD Error**：
   - 定义加编号
   - 三重角色预告（Critic 训练信号、最简优势函数、GAE 基础）
5. **关键洞察段落**：MC→TD 的演进在策略空间的再现（REINFORCE = MC for policy, Actor-Critic = TD for policy）
6. **对比表**（加粗列名）：

   |                    | DP  | MC  | TD  |
   | ------------------ | --- | --- | --- |
   | **需要模型？**     | 是  | 否  | 否  |
   | **需要完整轨迹？** | 否  | 是  | 否  |
   | **自举**           | 是  | 否  | 是  |
   | **偏差**           | 无  | 无  | 有  |
   | **方差**           | 低  | 高  | 中  |

7. **参考文献**：Sutton (1988)、Sutton & Barto Ch5-6

---

### 3.5 value-q.md（路线一：Q(s,a)）

**当前：** ~1500 字
**目标：** ~2500 字

具体增加条目：

1. **Q 的贝尔曼方程**（当前缺失）：
   - $Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s',a')$
   - 与 V 的贝尔曼方程对比
2. **Q\* 的贝尔曼最优方程**：
   - $Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s',a')$
   - 注意 max 出现在 Q 自身的更新中
3. **V 和 Q 的等价性**：
   - 知道 V* 可以推出 Q*（需要一步展开）
   - 知道 Q* 可以直接推出 π*
   - 但实际上 Q\* 更好用（不需要知道 P）
4. **off-policy 的预告**：
   - Q-Learning 学的是 Q\*（最优），但可以用任何策略收集数据
   - 这和 MC/REINFORCE 的 on-policy 形成对比
5. **参考文献**：Watkins & Dayan (1992)

---

### 3.6 policy-objective.md（路线二：J(θ)）

**当前：** ~1500 字（最薄的一节）
**目标：** ~2500 字

具体增加条目：

1. **连续动作空间的正式讨论**（当前缺失）：
   - 离散动作：argmax Q 可行（动作有限）
   - 连续动作：argmax Q 需要对连续函数求全局最大——不可行
   - 例子：机械臂 6 个关节各有连续力矩 → Q 函数输入是 6+1=7 维
2. **策略参数化的讨论**：
   - θ 是什么？神经网络的权重
   - 参数化如何改变搜索空间？从"选动作"变成"调参数"
   - 与非参数策略的对比
3. **J(θ) 的不同定义**：
   - episode 起点：$J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \gamma^t r_t]$
   - 平均奖励：$J(\theta) = \lim_{T\to\infty} \frac{1}{T} \mathbb{E}[\sum_{t=1}^T r_t]$（一句话提及）
4. **策略梯度的直觉**（预告，不深入推导）：
   - "好结果 → 加强沿途动作的概率"
   - "坏结果 → 削弱沿途动作的概率"
   - 这就是策略梯度定理的直觉（Ch5 展开推导）
5. **两条路线的正式对比表**（加粗列名，比当前版本更详细）

---

### 3.7 panorama.md（全景地图）

**当前：** ~2000 字
**目标：** ~2500 字

具体增加条目：

1. **探索-利用与两条路线的正式关联**：
   - 路线一的探索困境：ε-贪婪是人工补丁
   - 路线二的探索优势：随机性策略天然探索
   - 形式化：确定性策略 π(s)=argmax Q 完全没有探索能力
2. **Deep RL 的历史背景**：
   - TD-Gammon (1992)：TD 方法 + 神经网络的早期成功
   - DQN (2015)：深度 RL 的里程碑
   - 当前：LLM + RL 的结合（RLHF、GRPO）
3. **章节小结改为要点列表**（不用叙事体）

---

## Ch4：Value-Based 方法——Q-Learning 与深度 Q 网络

### 各节增厚

#### intro.md

**增加：**

- DeepMind 2013→2015 的历史叙事（arXiv → Nature，更学术的版本）
- Mnih et al. (2015) 引用带 DOI
- Value-Based 方法的正式定义段落
- 导航表加粗列名

#### q-learning.md

**增加：**

- Q-Learning 的收敛性讨论：在表格情况下保证收敛到 Q\*（Watkins & Dayan 1992 的结论）
- Decaying ε 的说明：ε 从 1.0 线性衰减到 0.01，为什么比固定 ε 更好
- GridWorld 的 Q 值热力图概念（描述，不需要实际图）
- on-policy (SARSA) vs off-policy (Q-Learning) 的一句话对比

#### from-q-to-dqn.md（现有文件，检查后可能需要调整）

- 确保不与 q-learning.md 重复
- 从"表格装不下"自然过渡

---

## Ch5：Policy-Based 方法——策略梯度与 REINFORCE

### 各节增厚

#### intro.md

**增加：**

- 连续动作空间的数学描述：$A \subseteq \mathbb{R}^d$，为什么 argmax 不可行
- 策略梯度方法的历史：Williams (1992) REINFORCE → Barto et al. → Schulman et al. (2017) PPO
- "本章暗线"改为更学术的表达："方差缩减（Variance Reduction）是本章的核心方法论线索"
- 导航表加粗列名

#### dice-game.md（现有文件）

**增加：**

- 策略网络的结构图（输入 → Linear → Softmax → 输出概率）
- 训练过程的数学解释：loss = -log π(a) · R 等价于梯度上升 ∇θ J
- 与 Q-Learning 的对比：Q-Learning 学"哪个好"，策略梯度学"怎么选"
- Williams (1992) 引用

#### policy-gradient.md（现有文件）

**增加：**

- 策略梯度定理的完整表述（不只是推导，还有结论框）
- 对数导数技巧（log-derivative trick）的单独解释：
  $\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$
- 回扣 Ch3：REINFORCE = MC for policy（用 G_t，需要完整轨迹，方差大）
- REINFORCE 的伪代码算法框
- Sutton et al. (2000) 引用

#### baseline-experiment.md

**增加：**

- 方差缩减的数学解释：$\text{Var}[\nabla J] = \text{Var}[G_t \cdot \nabla \log \pi] > \text{Var}[(G_t - b) \cdot \nabla \log \pi]$
- 为什么 V(s) 是最优基线（Williams 1992 的结论）
- 练习题保留
- 参考文献：Williams (1992)、Sutton et al. (2000)

---

## Ch6：Actor-Critic 架构

### 6.0 intro.md

**增加：**

- Actor-Critic 的历史：Witten (1977) 最早提出，Barto et al. (1983) 系统化
- 与 Ch4/Ch5 的定位对比表

### 6.1 advantage-function.md

**内容：**

1. 从 Ch5 基线实验承接：减掉 V(s) → 优势函数
2. A(s,a) = Q(s,a) - V(s) 的正式定义
3. 用 TD Error 近似 A：A ≈ r + γV(s') - V(s) = δ
4. 为什么 A 比 G_t 好（方差分析）
5. 棋类类比（保留）
6. 下棋的具体数值例子：V(s)=60%, Q(s,出车)=75%, A=15%
7. 公式编号

### 6.2 critic-training.md

**内容：**

1. Critic = V(s) 的神经网络实现
2. 网络结构图（状态 → MLP → V(s) 标量）
3. DP 方法展开：
   - 策略评估（Policy Evaluation）算法
   - 策略改进（Policy Improvement）的概念
   - 策略迭代（Policy Iteration）：评估 → 改进 → 评估 → ... 循环
4. MC 方法展开：
   - 用 G*t 更新 V：$L*{\text{Critic}} = (G*t - V*\phi(s))^2$
   - 无偏但方差大
5. TD 方法展开：
   - 用 δ 更新 V：$L_{\text{Critic}} = (r + \gamma V_\phi(s') - V_\phi(s))^2$
   - 有偏但方差低 → 实际首选
6. 三种方法的对比表（加粗列名）
7. 完整训练流程（编号步骤）
8. 参考文献：Sutton (1988)、Mnih et al. (2016) A3C

### 6.3 actor-critic.md

**内容：**

1. 从 REINFORCE 到 Actor-Critic 的关键步骤
2. Actor + Critic 的架构图
3. 完整训练循环：
   - Actor 选动作 → 环境返回 (r, s') → Critic 算 δ → 用 δ 更新 Actor → 用 δ² 更新 Critic
4. 与纯 REINFORCE 的对比表
5. 收敛性讨论：Actor-Critic 的收敛条件
6. A2C / A3C 的一句话预告（多个 actor 并行）
7. 参考文献：Barto et al. (1983)、Mnih et al. (2016) A3C

### 6.4 alphago.md（现有文件，检查衔接）

---

## Ch7：PPO——稳定训练的艺术

### intro.md

**增加：**

- 策略崩溃的具体例子：学习率太大 → 一步更新 → 策略急剧变差 → 从此恢复不了
- Schulman et al. (2017) PPO 论文引用
- TRPO → PPO 的演进关系（一句话）
- 导航表加粗列名

### 其他文件（现有内容）

- 检查交叉引用
- 检查语气

---

## 全局格式规范

### 公式编号

所有主要公式加编号，格式：(章节.序号)，如 (3.1)、(4.2)

### 参考文献格式

每章末尾统一格式：

```
## 参考文献
[^1]: Author, A. (Year). Title. Journal. [DOI/链接](url)
```

### 中英术语对照

首次出现时标注英文，后续不再重复。格式：中文（English）

### 表格规范

- 列名加粗
- 第一列加粗

### 衔接语句规范

- 节首：直接陈述，不用"上一节我们……"
- 节尾：不用"准备好了吗？"，用"下一节将讨论……"
