<div align="center">
  <h1>Hands-On Modern RL</h1>
  <p><strong>现代强化学习实战</strong></p>
  <p><em>From Code to Theory — 从代码到原理</em></p>

  <p>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/stargazers"><img src="https://img.shields.io/badge/Stars-0-eab676?style=for-the-badge&logo=github" alt="Stars" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/network/members"><img src="https://img.shields.io/badge/Forks-0-87a96b?style=for-the-badge&logo=github" alt="Forks" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/issues"><img src="https://img.shields.io/badge/Issues-0-c780e8?style=for-the-badge&logo=github" alt="Issues" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/pulls"><img src="https://img.shields.io/badge/Pull%20Requests-0-7ea9e1?style=for-the-badge&logo=github" alt="Pull Requests" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-e26d5c?style=for-the-badge" alt="CC BY-NC-SA 4.0 License" /></a>
  </p>

  <p>
    <a href="#课程概述">课程概述</a> &middot;
    <a href="#课程大纲">课程大纲</a> &middot;
    <a href="#本地运行环境">本地运行环境</a> &middot;
    <a href="#参与贡献">参与贡献</a>
  </p>
</div>

---

## 课程概述

强化学习（Reinforcement Learning）是机器学习中一个核心但门槛较高的分支。传统的教学路径通常从马尔可夫决策过程（MDP）和贝尔曼方程的形式化定义出发，这对许多学习者的耐心和理解力构成了不必要的挑战。

本课程采用另一种路径：**先建立直觉，再引入形式化**。我们认为，当学习者亲手实现一个算法、观察到它的行为，再回过头理解其背后的数学结构时，学习过程会更加自然，理解也会更加持久。

具体而言，课程围绕以下四条线索展开：

1. **代码先行**——第一章即动手训练 CartPole 智能体，第二章用 DPO 微调大语言模型，通过实验建立对状态、动作、奖励、策略的直觉。
2. **算法拆解**——逐行精读 PPO、DPO、GRPO 等工业级算法的实现，追问每一步设计选择背后的动机，而非仅仅调用 API。
3. **理论回溯**——在实践基础上系统讲授 MDP、贝尔曼方程、策略梯度定理、GAE 等理论，完成从直觉到形式化的闭环。
4. **前沿衔接**——从 RLHF 三阶段流水线（SFT → RM → PPO）到 RLAIF 与 Self-Play 等替代方案，从 VLM 的视觉问答强化训练到 Agentic RL 的多轮工具调用与信用分配，使学习者理解从"实验室 CartPole"到"工业级 LLM 后训练"之间的完整路径。

## 课程目标

学完本课程后，学习者应能够：

- 理解强化学习的核心数学框架（MDP、价值函数、策略梯度），并能够用代码将其实现。
- 阅读并理解 PPO、DPO、GRPO 等主流算法的原始论文及其工程实现。
- 理解强化学习在大语言模型后训练（post-training）中的核心角色，涵盖 RLHF 三阶段流水线（SFT → Reward Model → PPO）、离线对齐方法族（DPO / KTO / SimPO / IPO）的理论等价性、可验证奖励范式（RLVR）以及 GRPO / DAPO 等无 Critic 算法的工程实践。
- 针对给定的实际问题，合理选择并调试试用的强化学习算法。

## 课程大纲

### Part 1: 快速入门

| 章节           | 课题                                                                 | 核心内容                                                                                                                         |
| :------------- | :------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| **Chapter 01** | [RL 初印象：求解 CartPole](docs/chapter01_cartpole/intro.md)         | 运行第一个 CartPole 训练脚本，观察 reward 曲线从低到高的变化过程；在实验中理解状态、动作、奖励、策略等基本要素的含义。           |
| **Chapter 02** | [现代 RL 初体验：大语言模型与 DPO 对齐](docs/chapter02_dpo/intro.md) | 用 DPO 算法对 Qwen2.5-0.5B 进行偏好微调，直观体验后训练（post-training）的完整流程；理解预训练、SFT、RL 三阶段各自的定位与局限。 |

### Part 2: 理论与方法

| 章节           | 课题                                                               | 核心内容                                                                                                                                                                                                |
| :------------- | :----------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Chapter 03** | [MDP 与大模型语境](docs/chapter03_mdp/intro.md)                    | 通过两台老虎机（Bandit）引入 MDP 形式化定义；推导贝尔曼方程，建立价值函数与 TD Error 的直觉；完成从表格方法到神经网络函数逼近的过渡。                                                                   |
| **Chapter 04** | [深度强化学习 DQN](docs/chapter04_dqn/intro.md)                    | 从经典 Q-Learning 出发，引入深度 Q 网络；通过 CartPole、Atari、ViZDoom 与 stable-retro 宝可梦实战检验算法边界；解析经验回放与目标网络两大支柱机制；延伸至 Double DQN、Dueling DQN、Rainbow 等改进谱系。 |
| **Chapter 05** | [策略梯度与 Actor-Critic](docs/chapter05_policy_gradient/intro.md) | 从摇骰子实验出发推导策略梯度定理；实现 REINFORCE 算法并观察高方差问题；引入基线与优势函数，构建 Actor-Critic 架构。                                                                                     |
| **Chapter 06** | [PPO 与奖励模型](docs/chapter06_ppo/intro.md)                      | 深入剖析 PPO 的裁剪机制、GAE（广义优势估计）；理解 Reward Model 的训练方式；建立 PPO 与 LLM 对齐的对应关系。                                                                                            |

### Part 3: LLM 时代

| 章节           | 课题                                                               | 核心内容                                                                                                                                                                                                                                                                                    |
| :------------- | :----------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Chapter 07** | [对齐方法族：DPO / KTO / SimPO](docs/chapter07_alignment/intro.md) | 推导 Bradley-Terry 偏好模型到 DPO 损失函数的等价变换；对比 DPO、KTO、SimPO、IPO 等离线对齐方法的原理与适用场景。                                                                                                                                                                            |
| **Chapter 08** | [GRPO、DAPO 与 RLVR](docs/chapter08_grpo_rlvr/intro.md)            | 理解 GRPO 用组内相对比较替代 Critic 网络的核心思路；探讨 Pure-RL（如 DeepSeek-R1-Zero 范式）与可验证奖励范式 RLVR；延伸至 DAPO 等改进；探讨 RL Scaling 与 Test-time Scaling（Best-of-N、多数投票、MCTS）；过程监督奖励模型（PRM）与结果监督奖励模型（ORM）的原理、优劣与自动化 PRM 的探索。 |

### Part 4: 进阶与前沿

| 章节           | 课题                                                                 | 核心内容                                                                                                                                                                                                                                                                                                                   |
| :------------- | :------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chapter 09** | [连续动作控制 (SAC/TD3)](docs/chapter09_continuous_control/intro.md) | 在 PyBullet/MuJoCo 环境中对比高斯策略与确定性策略；实现 DDPG、TD3、SAC 等连续控制算法；介绍向量化环境与并行采样加速策略。                                                                                                                                                                                                  |
| **Chapter 10** | [RLHF 完整流水线](docs/chapter10_rlhf/intro.md)                      | 全景展示 LLM 后训练的 RLHF 三阶段流水线（SFT → RM → PPO）；训练数据工程（SFT 指令数据构造、偏好数据生成与 AI 交叉打分、数据清洗与闭环迭代）；奖励模型深入（Bradley-Terry 损失、Token-level / Step-level / Sequence-level 细粒度奖励、规则奖励与模型奖励的组合）；奖励黑客防范、训练稳定性、RLAIF 与 Self-Play 等前沿扩展。 |
| **Chapter 11** | [VLM 强化学习](docs/chapter11_vlm_rl/intro.md)                       | 用 GRPO 训练 VLM 回答视觉问题；讨论视觉 token 与文本 token 的奖励分配问题；介绍 VisPlay、VISTA-Gym 等代表性框架。                                                                                                                                                                                                          |
| **Chapter 12** | [Agentic RL](docs/chapter12_agentic_rl/intro.md)                     | 从单轮 RL 扩展到多轮工具调用与环境交互；理解动作空间扩展与信用分配；分析工具调用训练、多轮 MDP 建模与基础设施挑战。                                                                                                                                                                                                        |
| **Chapter 13** | [未来趋势](docs/chapter13_future_trends/intro.md)                    | Self-Play RL 与自进化闭环（经验回放与提炼、失败驱动的课程生成）；测试时计算扩展（inference-time search）；多模态与具身智能中的 RL 融合；多智能体系统（MARL）与 PettingZoo 实战；模型基 RL 等前沿方向。                                                                                                                     |

### 附录

| 附录           | 课题                                                           | 说明                                                                                                                                                                                                            |
| :------------- | :------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Appendix A** | [常见坑与解法](docs/appendix_common_pitfalls/intro.md)         | 策略崩溃、奖励黑客、显存溢出、训练不收敛四类常见故障的现象描述、理论解释与修复验证方法。                                                                                                                        |
| **Appendix B** | [工业级训练与评测](docs/appendix_industrial_training/intro.md) | 分布式训练架构（DP/TP/PP/EP 四大并行策略、混合精度训练 FP16/BF16/FP8、veRL / LLaMA-Factory / DeepSpeed / Megatron-LM 实践指南）；评测体系（Benchmark 选择指南、Badcase 分析四步法、自动化评测闭环与回归测试）。 |
| **Appendix C** | [算法速查](docs/appendix_algorithm_guide/intro.md)             | 算法选型决策矩阵与速查表，按任务场景给出推荐算法与理由。                                                                                                                                                        |

## 本地运行环境

本课程的文档站点基于 [VitePress](https://vitepress.dev/) 构建。在本地预览或贡献内容，需要以下步骤：

### 前置要求

- Node.js >= 18.0.0

### 安装与启动

```bash
git clone https://github.com/walkinglabs/hands-on-modern-rl.git
cd hands-on-modern-rl
npm install
npm run dev
```

启动后访问 `http://localhost:5173` 即可浏览课程内容。

### 常用命令

```bash
npm run build    # 构建静态站点
npm run preview  # 本地预览构建产物
npm run verify   # 构建检查（提交 PR 前请务必执行）
npm run format   # 代码格式化
npm run lint     # 代码规范检查
```

## 仓库结构

```text
hands-on-modern-rl/
├── docs/                      # 课程文档（VitePress）
│   ├── .vitepress/            # 站点配置与自定义组件
│   ├── public/                # 静态资源
│   ├── chapter*/              # 各章节 Markdown 文件
│   └── appendix*/             # 附录与补充材料
├── scripts/                   # 自动化脚本（站点地图生成等）
├── .github/workflows/         # CI/CD 配置
├── package.json               # 项目配置
└── AGENTS.md                  # 仓库维护规则
```

## 参与贡献

欢迎通过 Pull Request 或 Issue 参与本课程的改进。为保证协作效率，请注意以下几点：

1. **保持单一职责**：每个 PR 聚焦于一个明确的改动，避免混合不相关的修改。
2. **遵循目录命名规范**：在 `docs/` 下新增内容时，使用连字符命名法（如 `chapter16-new-topic`），并以 `index.md` 作为入口文件。
3. **提交前验证**：若修改涉及 `.vitepress/config.mjs` 或构建脚本，请运行 `npm run verify` 确认无破坏性变更。
4. **Commit 规范**：遵循 [Conventional Commits](https://www.conventionalcommits.org/)（如 `feat:`, `fix:`, `docs:`, `chore:`）。

详细的维护与协作规则参见 [AGENTS.md](./AGENTS.md)。

## 许可协议

本课程内容基于 [CC BY-NC-SA 4.0（署名-非商业性使用-相同方式共享 4.0 国际）](./LICENSE) 协议发布。

---

<div align="center">
  <sub>Maintained by WalkingLabs and the open-source community.</sub>
</div>
