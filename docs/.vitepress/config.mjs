/* global process */
import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { createRequire } from 'module'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const require = createRequire(import.meta.url)
const markdownItFootnote = require('markdown-it-footnote')

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const packageJsonPath = path.resolve(__dirname, '../../package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))

const isVercel = process.env.VERCEL === '1' || !!process.env.VERCEL_URL

function parseRepository() {
  const repositoryUrl =
    process.env.GITHUB_REPOSITORY ||
    packageJson.repository?.url ||
    packageJson.repository ||
    ''

  if (repositoryUrl.includes('/')) {
    if (repositoryUrl.includes(':')) {
      const sshMatch = repositoryUrl.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
      if (sshMatch) {
        return { owner: sshMatch[1], repo: sshMatch[2] }
      }
    }

    const normalized = repositoryUrl
      .replace(/^https?:\/\/github\.com\//, '')
      .replace(/^git@github\.com:/, '')
      .replace(/\.git$/, '')

    if (normalized.includes('/')) {
      const [owner, repo] = normalized.split('/')
      return { owner, repo }
    }
  }

  return { owner: 'walkinglabs', repo: packageJson.name || 'course-template' }
}

const { owner, repo } = parseRepository()
const base = process.env.BASE || (isVercel ? '/' : `/${repo}/`)
const siteUrl = process.env.SITE_URL || `https://${owner}.github.io/${repo}`
const editLinkPattern = `https://github.com/${owner}/${repo}/edit/main/docs/:path`

const zhNav = [
  { text: '前言', link: '/preface/intro' },
  { text: '极速入门', link: '/chapter01_cartpole/intro' },
  { text: '理论与方法', link: '/chapter03_mdp/intro' },
  { text: 'LLM时代', link: '/chapter07_alignment/intro' },
  { text: '进阶应用与前沿', link: '/chapter09_continuous_control/intro' }
]

const enNav = [
  { text: 'Start', link: '/en/guide/getting-started' },
  {
    text: 'Demo Course',
    items: [
      { text: 'Course Tour', link: '/en/demo/' },
      { text: '01 Positioning and Map', link: '/en/demo/chapter-01/' },
      { text: '02 Content and Practice', link: '/en/demo/chapter-02/' },
      { text: '03 Release and Delivery', link: '/en/demo/chapter-03/' }
    ]
  },
  { text: 'Deployment', link: '/en/deployment/' }
]

const zhSidebar = {
  '/': [
    {
      text: '前言',
      items: [
        { text: '前言', link: '/preface/intro' },
        { text: '强化学习简史', link: '/preface/brief-history' }
      ]
    },
    {
      text: 'Part 1: 极速入门',
      items: [
        {
          text: '第1章：RL初印象',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter01_cartpole/intro' },
            { text: '核心原理', link: '/chapter01_cartpole/principles' },
            { text: '训练与指标', link: '/chapter01_cartpole/metrics' }
          ]
        },
        {
          text: '第2章：现代RL初体验',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter02_dpo/intro' },
            { text: '核心原理', link: '/chapter02_dpo/principles' },
            { text: '训练与指标', link: '/chapter02_dpo/metrics' }
          ]
        }
      ]
    },
    {
      text: 'Part 2: 理论与方法',
      items: [
        {
          text: '第3章：MDP 与大模型语境',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter03_mdp/intro' },
            { text: '动手：两台老虎机', link: '/chapter03_mdp/bandit' },
            { text: 'MDP 形式化与价值函数', link: '/chapter03_mdp/formalism' },
            {
              text: '贝尔曼方程与 TD Error',
              link: '/chapter03_mdp/bellman-equation'
            },
            { text: '经典方法与路线图', link: '/chapter03_mdp/classic-methods' }
          ]
        },
        {
          text: '第4章：深度强化学习 DQN',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter04_dqn/intro' },
            {
              text: '从 Q-Learning 到 DQN',
              link: '/chapter04_dqn/from-q-to-dqn'
            },
            { text: 'DQN 三大组件', link: '/chapter04_dqn/dqn-components' },
            {
              text: '动手：DQN 玩 CartPole',
              link: '/chapter04_dqn/cartpole-dqn'
            },
            {
              text: '动手：从像素学玩 Atari',
              link: '/chapter04_dqn/atari-dqn'
            },
            {
              text: '动手：3D 第一人称 ViZDoom',
              link: '/chapter04_dqn/vizdoom-dqn'
            },
            {
              text: '动手：stable-retro 玩宝可梦',
              link: '/chapter04_dqn/retro-pokemon'
            },
            { text: '观察训练过程', link: '/chapter04_dqn/training-analysis' },
            { text: 'DQN 家族与视角迁移', link: '/chapter04_dqn/dqn-family' }
          ]
        },
        {
          text: '第5章：策略梯度与 Actor-Critic',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter05_policy_gradient/intro' },
            {
              text: '动手：摇骰子赌博机',
              link: '/chapter05_policy_gradient/dice-game'
            },
            {
              text: '策略梯度定理与 REINFORCE',
              link: '/chapter05_policy_gradient/policy-gradient'
            },
            {
              text: 'Actor-Critic 架构',
              link: '/chapter05_policy_gradient/actor-critic'
            },
            {
              text: '基线实验与总结',
              link: '/chapter05_policy_gradient/baseline-experiment'
            }
          ]
        },
        { text: '第6章：PPO 与奖励模型', link: '/chapter06_ppo/intro' }
      ]
    },
    {
      text: 'Part 3: LLM 时代',
      items: [
        {
          text: '第7章：对齐方法族（DPO / KTO / SimPO）',
          link: '/chapter07_alignment/intro'
        },
        {
          text: '第8章：GRPO、DAPO 与 RLVR',
          link: '/chapter08_grpo_rlvr/intro'
        }
      ]
    },
    {
      text: 'Part 4: 进阶与前沿',
      items: [
        {
          text: '第9章：连续动作控制 (SAC/TD3)',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter09_continuous_control/intro' },
            {
              text: '动手：PyBullet 机器人仿真',
              link: '/chapter09_continuous_control/pybullet-hands-on'
            }
          ]
        },
        { text: '第10章：RLHF 完整流水线', link: '/chapter10_rlhf/intro' },
        { text: '第11章：VLM 强化学习', link: '/chapter11_vlm_rl/intro' },
        { text: '第12章：Agentic RL', link: '/chapter12_agentic_rl/intro' },
        {
          text: '第13章：未来趋势',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter13_future_trends/intro' },
            {
              text: '动手：PettingZoo 多智能体',
              link: '/chapter13_future_trends/pettingzoo'
            }
          ]
        }
      ]
    },
    {
      text: '附录',
      items: [
        {
          text: '附录A：常见坑与解法',
          link: '/appendix_common_pitfalls/intro'
        },
        {
          text: '附录B：工业级训练与评测',
          link: '/appendix_industrial_training/intro'
        },
        { text: '附录C：算法速查', link: '/appendix_algorithm_guide/intro' }
      ]
    }
  ]
}

const enSidebar = {
  '/en/guide/': [
    {
      text: 'Template Guide',
      items: [
        { text: 'Getting Started', link: '/en/guide/getting-started' },
        { text: 'Course Structure', link: '/en/guide/course-structure' },
        { text: 'Project Setup Checklist', link: '/en/guide/project-setup' },
        { text: 'Deployment Guide', link: '/en/deployment/' }
      ]
    }
  ],
  '/en/demo/': [
    {
      text: 'Demo Course Tour',
      items: [
        { text: 'Overview', link: '/en/demo/' },
        { text: 'Learning Path', link: '/en/demo/#learning-path' },
        { text: 'Deliverables', link: '/en/demo/#stage-deliverables' }
      ]
    },
    {
      text: 'Part I. Shape the Course',
      collapsed: false,
      items: [
        {
          text: 'Chapter 01 Positioning and Map',
          link: '/en/demo/chapter-01/'
        },
        {
          text: 'Chapter 01 Advanced Notes',
          link: '/en/demo/chapter-01/advanced'
        }
      ]
    },
    {
      text: 'Part II. Write Content and Practice',
      collapsed: false,
      items: [
        {
          text: 'Chapter 02 Content and Practice',
          link: '/en/demo/chapter-02/'
        },
        {
          text: 'Chapter 02 Homework Brief',
          link: '/en/demo/chapter-02/homework/'
        }
      ]
    },
    {
      text: 'Part III. Release and Delivery',
      collapsed: false,
      items: [
        {
          text: 'Chapter 03 Release and Delivery',
          link: '/en/demo/chapter-03/'
        }
      ]
    }
  ],
  '/en/': [
    {
      text: 'Start Here',
      items: [
        { text: 'Home', link: '/en/' },
        { text: 'Getting Started', link: '/en/guide/getting-started' },
        { text: 'Demo Course', link: '/en/demo/' },
        { text: 'Deployment', link: '/en/deployment/' }
      ]
    }
  ]
}

export default withMermaid(
  defineConfig({
    lang: 'zh-CN',
    title: 'Course Template',
    description: 'WalkingLabs course template powered by VitePress',
    base,
    cleanUrls: true,
    lastUpdated: true,
    markdown: {
      math: true,
      config: (md) => {
        md.use(markdownItFootnote)

        // Workaround: markdown-it-mathjax3 may inject <style> tags that
        // Vue's DOM compiler rejects in dev mode.  Strip them from the
        // rendered output so the page compiles cleanly.
        const stripStyles = (html) =>
          html.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
        const origInline = md.renderer.rules.math_inline
        const origBlock = md.renderer.rules.math_block
        if (origInline) {
          md.renderer.rules.math_inline = function (...args) {
            return stripStyles(origInline.apply(this, args))
          }
        }
        if (origBlock) {
          md.renderer.rules.math_block = function (...args) {
            return stripStyles(origBlock.apply(this, args))
          }
        }
      }
    },
    ignoreDeadLinks: true,
    head: [
      ['link', { rel: 'icon', href: `${base}favicon.svg` }],
      ['meta', { name: 'theme-color', content: '#0f766e' }],
      [
        'meta',
        { name: 'viewport', content: 'width=device-width, initial-scale=1.0' }
      ],
      ['meta', { name: 'author', content: 'WalkingLabs' }],
      ['meta', { name: 'robots', content: 'index,follow' }],
      ['meta', { property: 'og:title', content: 'Course Template' }],
      [
        'meta',
        {
          property: 'og:description',
          content:
            'A reusable bilingual documentation and deployment template for future WalkingLabs courses'
        }
      ],
      ['meta', { property: 'og:type', content: 'website' }],
      ['meta', { property: 'og:url', content: siteUrl }],
      ['meta', { name: 'twitter:card', content: 'summary_large_image' }]
    ],
    locales: {
      zh: {
        label: '简体中文',
        lang: 'zh-CN',
        link: '/zh/',
        title: 'Course Template',
        description: 'WalkingLabs 双语课程仓库模板',
        themeConfig: {
          nav: zhNav,
          sidebar: zhSidebar,
          editLink: {
            pattern: editLinkPattern,
            text: '在 GitHub 上编辑此页'
          },
          footer: {
            message: '为可复用的双语课程交付而构建',
            copyright: 'Copyright © WalkingLabs'
          },
          outline: {
            level: [2, 3],
            label: '页面目录'
          },
          lastUpdated: {
            text: '最后更新'
          },
          docFooter: {
            prev: '上一页',
            next: '下一页'
          },
          darkModeSwitchLabel: '外观',
          lightModeSwitchTitle: '切换到浅色模式',
          darkModeSwitchTitle: '切换到深色模式',
          sidebarMenuLabel: '菜单',
          returnToTopLabel: '返回顶部',
          langMenuLabel: '切换语言',
          skipToContentLabel: '跳转到正文',
          notFound: {
            title: '页面未找到',
            quote: '这个地址不存在，试试从中文首页重新进入。',
            link: '/zh/',
            linkText: '返回中文首页',
            linkLabel: '返回中文首页'
          }
        }
      },
      en: {
        label: 'English',
        lang: 'en-US',
        link: '/en/',
        title: 'Course Template',
        description: 'WalkingLabs bilingual course repository template',
        themeConfig: {
          nav: enNav,
          sidebar: enSidebar,
          editLink: {
            pattern: editLinkPattern,
            text: 'Edit this page on GitHub'
          },
          footer: {
            message: 'Built for reusable bilingual course delivery',
            copyright: 'Copyright © WalkingLabs'
          },
          outline: {
            level: [2, 3],
            label: 'On this page'
          },
          lastUpdated: {
            text: 'Last updated'
          },
          docFooter: {
            prev: 'Previous page',
            next: 'Next page'
          },
          darkModeSwitchLabel: 'Appearance',
          lightModeSwitchTitle: 'Switch to light theme',
          darkModeSwitchTitle: 'Switch to dark theme',
          sidebarMenuLabel: 'Menu',
          returnToTopLabel: 'Return to top',
          langMenuLabel: 'Change language',
          skipToContentLabel: 'Skip to content',
          notFound: {
            title: 'Page not found',
            quote:
              'This page is missing. Try jumping back in from the English home page.',
            link: '/en/',
            linkText: 'Take me to English home',
            linkLabel: 'Go to English home'
          }
        }
      }
    },
    themeConfig: {
      logo: '/logo.svg',
      siteTitle: 'Hands-on Modern RL',
      nav: zhNav,
      sidebar: zhSidebar,
      socialLinks: [
        { icon: 'github', link: `https://github.com/${owner}/${repo}` }
      ],
      search: {
        provider: 'local'
      },
      editLink: {
        pattern: editLinkPattern,
        text: 'Edit this page on GitHub'
      },
      footer: {
        message: 'Built for reusable bilingual course delivery',
        copyright: 'Copyright © WalkingLabs'
      },
      outline: {
        level: [2, 3],
        label: 'On this page'
      }
    }
  })
)
