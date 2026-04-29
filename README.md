<div align="center">
  <h1>Hands-On Modern RL</h1>
  <p><strong>现代强化学习实战课程</strong></p>
  <p><em>Practice-first reinforcement learning, from CartPole to LLM post-training and agentic systems.</em></p>

  <p>
    <a href="https://walkinglabs.github.io/hands-on-modern-rl/"><img src="https://img.shields.io/badge/Course-Online-2563eb?style=flat-square" alt="Online Course" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/actions/workflows/deploy.yml"><img src="https://github.com/walkinglabs/hands-on-modern-rl/actions/workflows/deploy.yml/badge.svg" alt="Deploy VitePress site to Pages" /></a>
    <a href="https://github.com/walkinglabs/hands-on-modern-rl/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-111827?style=flat-square" alt="CC BY-NC-SA 4.0 License" /></a>
    <img src="https://img.shields.io/badge/Node-%3E%3D18-16a34a?style=flat-square" alt="Node >= 18" />
    <img src="https://img.shields.io/badge/Docs-VitePress-646cff?style=flat-square" alt="VitePress" />
  </p>

  <p>
    <a href="#overview">Overview</a> ·
    <a href="#curriculum">Curriculum</a> ·
    <a href="#labs">Labs</a> ·
    <a href="#getting-started">Getting Started</a> ·
    <a href="#contributing">Contributing</a>
  </p>
</div>

---

## Overview

**Hands-On Modern RL** 是一门面向现代强化学习实践的开放课程。它从可运行的代码和可观察的训练现象出发，逐步回到强化学习的数学结构，再连接到大语言模型后训练、偏好对齐、可验证奖励、视觉语言模型和 Agentic RL 等前沿主题。

这不是一份只罗列公式的讲义，也不是一组只调用框架 API 的 demo。课程的目标是建立一条完整路径：先让学习者看到智能体如何在环境中试错、如何从奖励中改进行为，再解释这些现象背后的状态、动作、价值函数、策略梯度、优势估计、奖励建模和信用分配。

课程内容采用中英双语术语体系组织，主文本以中文讲解为主，保留必要英文术语、论文名和工程概念，方便读者阅读原始论文、开源实现和工业系统文档。

## Status

This repository is an active courseware project. The material is being expanded and refined chapter by chapter, with emphasis on correctness, runnable examples, and a stable learning path.

- Course site: [walkinglabs.github.io/hands-on-modern-rl](https://walkinglabs.github.io/hands-on-modern-rl/)
- Source content: [`docs/`](docs/)
- Runnable examples: [`code/`](code/)
- Local verification: `npm run verify`
- License: [CC BY-NC-SA 4.0](LICENSE)

Issues and pull requests are welcome for typo fixes, conceptual corrections, reproducibility improvements, additional references, and carefully scoped course extensions.

## Design Principles

The course is organized around a few engineering and teaching principles:

1. **Practice before formalism.** Each major topic starts from an experiment, metric, failure case, or implementation detail before introducing the mathematical abstraction.
2. **Theory that explains behavior.** MDPs, Bellman equations, policy gradients, GAE, PPO clipping, DPO objectives, and GRPO-style group advantages are introduced as tools for explaining what the code is doing.
3. **Modern RL, not only classic RL.** The curriculum covers classic control and deep RL, then moves into RLHF, preference optimization, RLVR, VLM RL, and multi-turn agent training.
4. **Debuggability as a first-class skill.** Training collapse, reward hacking, KL drift, entropy decay, OOM failures, and evaluation blind spots are treated as core material rather than afterthoughts.
5. **Readable systems over black boxes.** The code examples favor explicit implementation, inspectable metrics, and clean experiment boundaries so learners can modify and extend them.

## Audience

This course is designed for learners who want to understand reinforcement learning by building and inspecting working systems.

It is especially useful for:

- machine learning engineers moving from supervised learning into RL;
- researchers and students preparing to read modern RL and alignment papers;
- LLM practitioners who want to understand RLHF, DPO, GRPO, RLVR, and post-training systems;
- builders of tool-using agents, Web agents, Code agents, and evaluation pipelines;
- self-directed learners who prefer code, experiments, and visual intuition before dense formalism.

Recommended background:

- Python programming experience;
- basic PyTorch familiarity;
- linear algebra, probability, and calculus at an introductory ML level;
- comfort reading papers and tracing open-source training scripts.

The mathematical appendix is included for review, so the course does not assume perfect fluency from day one.

## Learning Outcomes

After completing the course, a learner should be able to:

- implement and explain the core reinforcement learning loop: environment interaction, rollout collection, reward feedback, policy update, and evaluation;
- connect MDPs, value functions, Bellman equations, TD learning, policy gradients, and advantage estimates to concrete training behavior;
- read and modify DQN, REINFORCE, Actor-Critic, PPO, DPO, GRPO, and related implementations;
- reason about LLM post-training pipelines, including SFT, reward modeling, PPO-style RLHF, DPO-family methods, and verifiable-reward training;
- diagnose common RL failure modes such as reward hacking, policy collapse, KL instability, non-convergence, and evaluation leakage;
- design a reasonable algorithm and infrastructure plan for a new RL problem, including metrics, debugging strategy, and evaluation protocol.

## Curriculum

The course is divided into four parts plus appendices. The online site contains the full text, diagrams, code references, and chapter-level navigation.

### Preface

| Topic                                               | Description                                                        |
| :-------------------------------------------------- | :----------------------------------------------------------------- |
| [写在开头](docs/preface/intro.md)                   | Course positioning, learning path, and how to use the material.    |
| [强化学习简史](docs/preface/brief-history/index.md) | From trial-and-error learning to AlphaGo, RLHF, and LLM alignment. |

### Part I: Foundations by Experiment

| Chapter | Topic                                                                | Core Questions                                                                                                     |
| :------ | :------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| 01      | [RL 初印象：求解 CartPole](docs/chapter01_cartpole/intro.md)         | What are state, action, reward, policy, value, entropy, and training curves in a real environment?                 |
| 02      | [现代 RL 初体验：大语言模型与 DPO 对齐](docs/chapter02_dpo/intro.md) | How does preference optimization change model behavior, and what do loss, reward margin, and reward accuracy mean? |
| Summary | [Part I Summary](docs/summaries/part1-summary.md)                    | What intuition should be stable before entering the formal theory?                                                 |

### Part II: Core Theory and Algorithms

| Chapter | Topic                                                           | Core Questions                                                                                                      |
| :------ | :-------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| 03      | [MDP 与价值函数](docs/chapter03_mdp/intro.md)                   | How do bandits, MDPs, value functions, Bellman equations, and TD errors formalize sequential decision-making?       |
| 04      | [Q-Learning 与 DQN](docs/chapter04_dqn/intro.md)                | Why do replay buffers, target networks, CNN encoders, Double DQN, Dueling DQN, and Rainbow-style extensions matter? |
| 05      | [策略梯度与 REINFORCE](docs/chapter05_policy_gradient/intro.md) | How can a policy be optimized directly, and why do baselines reduce gradient variance?                              |
| 06      | [Actor-Critic 架构](docs/chapter06_actor_critic/intro.md)       | How do actors and critics divide the learning problem, and how does TD error become an advantage signal?            |
| 07      | [PPO：稳定训练的艺术](docs/chapter06_ppo/intro.md)              | How do clipping, trust-region intuition, GAE, and reward models make policy optimization more stable?               |
| Summary | [Part II Summary](docs/summaries/part2-summary.md)              | What algorithmic patterns recur across classic and modern RL?                                                       |

### Part III: Reinforcement Learning for LLMs

| Chapter | Topic                                                                  | Core Questions                                                                                                                   |
| :------ | :--------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| 08      | [RLHF 完整流水线](docs/chapter10_rlhf/intro.md)                        | How do instruction data, reward models, PPO training, RLAIF, and data cycles fit together?                                       |
| 09      | [对齐与推理强化：DPO / GRPO / RLVR](docs/chapter07_alignment/intro.md) | How do offline preference methods, implicit rewards, group-relative advantages, and verifiable rewards train reasoning behavior? |
| 10      | [Agentic RL](docs/chapter12_agentic_rl/intro.md)                       | How do multi-turn interaction, tool use, trajectory synthesis, and credit assignment change the RL problem?                      |
| Summary | [Part III Summary](docs/summaries/part3-summary.md)                    | What makes LLM RL different from classic environment RL?                                                                         |

### Part IV: Frontiers and Advanced Systems

| Chapter | Topic                                                                | Core Questions                                                                                                              |
| :------ | :------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- |
| 11      | [VLM 强化学习](docs/chapter11_vlm_rl/intro.md)                       | How do visual inputs, multimodal rewards, and VLM-specific evaluation change the training loop?                             |
| 12      | [连续动作算法与具身智能](docs/chapter09_continuous_control/intro.md) | How do DDPG, TD3, SAC, HER, diffusion policies, and simulation-to-real concerns appear in continuous control?               |
| 13      | [未来趋势](docs/chapter13_future_trends/intro.md)                    | How do test-time reasoning, multi-agent RL, model-based RL, offline RL, self-play, and embodied multimodal systems connect? |
| Summary | [Part IV Summary](docs/summaries/part4-summary.md)                   | What directions should learners track after finishing the core course?                                                      |

### Appendices

| Appendix | Topic                                                          | Description                                                                                                        |
| :------- | :------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| A        | [强化学习训练调试指南](docs/appendix_common_pitfalls/intro.md) | Failure modes, symptoms, root causes, and repair strategies for RL training.                                       |
| B        | [RL 工程实践指南](docs/appendix_industrial_training/intro.md)  | Sampling infrastructure, distributed training, monitoring, evaluation, badcase analysis, and industrial exercises. |
| C        | [算法选型与工程框架](docs/appendix_algorithm_guide/intro.md)   | Decision matrices for algorithm selection and framework choices.                                                   |
| D        | [强化学习经典项目](docs/appendix_game_projects/intro.md)       | Curated project references for expanding beyond the course examples.                                               |
| E        | [数学基础](docs/appendix_math/intro.md)                        | Linear algebra, probability, statistics, optimization, and information theory for RL.                              |

## Labs

The [`code/`](code/) directory contains runnable examples aligned with the chapters. Each chapter is intended to be small enough to inspect, run, and modify independently.

| Area                    | Code Path                                                                                                                                    | Representative Exercises                                                                  |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| Classic control         | [`code/chapter01_cartpole/`](code/chapter01_cartpole/)                                                                                       | Train CartPole, inspect rewards and episode length, compare PPO implementations.          |
| Preference tuning       | [`code/chapter02_dpo/`](code/chapter02_dpo/)                                                                                                 | Generate preference data, train with DPO, compare model behavior before and after tuning. |
| MDPs and value learning | [`code/chapter03_mdp/`](code/chapter03_mdp/)                                                                                                 | Run bandit policies, solve GridWorld, verify Bellman updates numerically.                 |
| Deep Q-learning         | [`code/chapter04_dqn/`](code/chapter04_dqn/)                                                                                                 | Implement replay buffers, target networks, and Double DQN variants.                       |
| Policy gradients        | [`code/chapter05_policy_gradient/`](code/chapter05_policy_gradient/)                                                                         | Compare REINFORCE, baseline variants, and Actor-Critic updates.                           |
| PPO                     | [`code/chapter06_ppo/`](code/chapter06_ppo/)                                                                                                 | Train LunarLander, inspect clipping, visualize GAE, and compare training stability.       |
| Alignment and RLVR      | [`code/chapter07_alignment/`](code/chapter07_alignment/), [`code/chapter08_grpo_rlvr/`](code/chapter08_grpo_rlvr/)                           | Explore DPO rewards, GRPO group advantages, and rule-based verifiable rewards.            |
| RLHF                    | [`code/chapter10_rlhf/`](code/chapter10_rlhf/)                                                                                               | Walk through SFT, reward model training, and PPO-style alignment.                         |
| VLM and agents          | [`code/chapter11_vlm_rl/`](code/chapter11_vlm_rl/), [`code/chapter12_agentic_rl/`](code/chapter12_agentic_rl/)                               | Build multimodal reward functions and tool-use agents.                                    |
| Advanced topics         | [`code/chapter09_continuous_control/`](code/chapter09_continuous_control/), [`code/chapter13_future_trends/`](code/chapter13_future_trends/) | Study continuous control, tree-of-thought search, and multi-agent RL.                     |

See [`code/README.md`](code/README.md) for the code index and per-chapter dependency notes.

## Recommended Study Path

A practical path through the repository:

1. Read the [course introduction](docs/preface/intro.md) and run the CartPole example.
2. Skim the DPO chapter early, even before the full theory, to anchor the LLM post-training motivation.
3. Work through Chapters 03-07 in order; this is the conceptual core.
4. Return to RLHF, DPO, GRPO, and RLVR with the policy-gradient and PPO machinery in place.
5. Use the debugging and engineering appendices whenever a training run behaves unexpectedly.
6. Treat the frontier chapters as extensions: VLM RL, Agentic RL, continuous control, multi-agent systems, and test-time reasoning.

## Getting Started

### Read Online

The published course site is available at:

```text
https://walkinglabs.github.io/hands-on-modern-rl/
```

### Run the Documentation Site Locally

Requirements:

- Node.js >= 18.0.0
- npm

```bash
git clone https://github.com/walkinglabs/hands-on-modern-rl.git
cd hands-on-modern-rl
npm install
npm run dev
```

Then open the local VitePress server shown in the terminal, typically:

```text
http://localhost:5173
```

### Verify the Site

Before opening a pull request that changes documentation structure, theme code, navigation, build scripts, or generated assets, run:

```bash
npm run verify
```

This checks formatting, lints the VitePress theme, builds the site, and verifies the expected build artifacts.

### Run the Course Code

Most code examples are Python-based and organized by chapter.

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For smaller installs, prefer a chapter-specific requirements file:

```bash
pip install -r chapter01_cartpole/requirements.txt
python chapter01_cartpole/1-ppo_cartpole.py
```

Some chapters may require additional system libraries, GPU support, model downloads, or environment-specific setup. Prefer starting with Chapter 01 before running LLM, VLM, or simulator-heavy examples.

## Repository Layout

```text
hands-on-modern-rl/
├── docs/                      # VitePress course content
│   ├── .vitepress/            # Site config, navigation, theme overrides
│   ├── public/                # Static assets copied to the built site
│   ├── preface/               # Course introduction and historical context
│   ├── chapter*/              # Main course chapters
│   ├── appendix*/             # Supplementary material and references
│   └── summaries/             # Part-level reviews and synthesis notes
├── code/                      # Runnable chapter-aligned examples
├── scripts/                   # Maintenance and verification scripts
├── .github/workflows/         # GitHub Pages deployment workflow
├── package.json               # Site scripts and dependencies
├── AGENTS.md                  # Repository maintenance guidelines
└── README.md                  # Project overview
```

## Development Commands

```bash
npm run dev           # Start local documentation server
npm run build         # Build the static site
npm run preview       # Preview the built site locally
npm run format        # Format repository files with Prettier
npm run format:check  # Check formatting
npm run lint          # Lint VitePress theme code
npm run verify        # Run format check, lint, build, and artifact checks
```

## Contributing

Contributions should make the course clearer, more correct, more reproducible, or easier to navigate.

Good contributions include:

- fixing conceptual errors, equations, diagrams, broken links, or typos;
- improving explanations without changing the intended learning path;
- adding small, reproducible experiments that clarify an existing chapter;
- improving scripts, build reliability, navigation, or accessibility;
- adding references to high-quality papers, official docs, or widely used open-source implementations.

Please keep pull requests focused. A good PR usually changes one chapter, one lab, one diagram group, or one infrastructure concern at a time.

When adding content:

1. Put course material under [`docs/`](docs/).
2. Use kebab-case for new directories and files.
3. Prefer directory-based routes with `index.md`.
4. Update [`docs/.vitepress/config.mjs`](docs/.vitepress/config.mjs) when adding navigable pages.
5. Run `npm run verify` before requesting review when the change touches config, theme, scripts, or generated site output.
6. Use Conventional Commits, for example `docs: clarify ppo clipping` or `fix: repair chapter link`.

For repository-specific maintenance rules, see [`AGENTS.md`](AGENTS.md).

## Citation

If you use this course in teaching material, study notes, or derivative non-commercial educational work, please cite the repository:

```bibtex
@misc{hands_on_modern_rl,
  title        = {Hands-On Modern RL: Practice-first reinforcement learning from CartPole to LLM post-training and agentic systems},
  author       = {WalkingLabs},
  year         = {2026},
  howpublished = {\url{https://github.com/walkinglabs/hands-on-modern-rl}},
  note         = {Open courseware repository}
}
```

## License

The course material is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).

You may share and adapt the material for non-commercial purposes, provided that appropriate credit is given and derivative work is distributed under the same license.

---

<div align="center">
  <sub>Maintained by WalkingLabs and contributors.</sub>
</div>
