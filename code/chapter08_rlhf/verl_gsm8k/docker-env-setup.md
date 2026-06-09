# veRL + GSM8K PPO：Docker 容器与环境搭建记录

本文档整理在 `hands-on-modern-rl` 容器中搭建 **verl 0.5.x + vLLM + flash_attn + torch 2.7.1** 环境的完整过程，用于跑 GSM8K PPO 训练。

> 适用场景：宿主机已有 GPU、模型目录 `/opt/data`、课程代码仓库，需要在隔离容器里复现训练环境。

---

## 1. 最终可用环境

| 组件 | 版本 | 说明 |
|------|------|------|
| 基础镜像 | `javirub/flashattention-pytorch:flashattn2.8.1-pytorch2.7.1-cuda12.8-cudnn9-runtime` | 预装 torch 2.7.1 + flash_attn 2.8.1 |
| torch | 2.7.1+cu128 | **不要被 pip 降级** |
| flash_attn | 2.8.1 | 与 torch 2.7.1 绑定 |
| verl | 0.5.0（`v0.5.x` 分支） | 安装路径 `/opt/verl` |
| vllm | **0.10.0**（`--no-deps` 安装） | 官方要求 torch==2.7.1，与镜像匹配 |
| ray | 2.47.1 | verl 依赖，训练脚本会自动启动 |
| transformers | 4.55.4 | 需保持 4.x，避免被升到 5.x |
| tokenizers | 0.21.4 | 与 transformers 4.x 配套 |

**关键结论：** verl 0.5.x 官方声明 vllm ≤ 0.8.5，但该区间要求 torch 2.6.0，会与镜像里的 flash_attn 冲突。实测 **`verl 0.5.0 + vllm 0.10.0 (--no-deps) + torch 2.7.1`** 可正常跑 PPO 训练。

---

## 2. 启动 Docker 容器

### 2.1 推荐启动命令（当前在用）

```bash
docker run -d \
  --name hands-on-modern-rl \
  --network host \
  --gpus all \
  --ipc=host \
  -v /root/hands-on-modern-rl:/root/hands-on-modern-rl \
  -v /opt/data:/opt/data \
  -w /root/hands-on-modern-rl \
  javirub/flashattention-pytorch:flashattn2.8.1-pytorch2.7.1-cuda12.8-cudnn9-runtime \
  sleep infinity
```

### 2.2 参数说明

| 参数 | 作用 |
|------|------|
| `--network host` | 容器与宿主机共享网络，Ray / wandb / 代理访问更方便 |
| `--gpus all` | 暴露全部 GPU |
| `--ipc=host` | 共享 IPC，多进程 / vLLM / FSDP 需要较大共享内存 |
| `-v .../hands-on-modern-rl` | 挂载课程代码，容器内改动会同步到宿主机 |
| `-v /opt/data:/opt/data` | 挂载模型与数据目录 |

### 2.3 进入容器

```bash
docker exec -it hands-on-modern-rl bash
```

### 2.4 镜像选型说明

最初尝试过 `nvidia/cuda:12.8.1-devel-ubuntu22.04` 裸镜像，但需要从零编译 flash_attn，耗时且易出错。后改用 **预装 flash_attn 的 PyTorch 镜像**，保留 torch 2.7.1 栈，再单独处理 vllm 兼容性。

---

## 3. 配置网络代理（可选）

容器内访问 GitHub / PyPI 需要代理时，在 shell 中设置：

```bash
export https_proxy=http://192.168.99.128:7890
export http_proxy=http://192.168.99.128:7890
export HTTP_PROXY=http://192.168.99.128:7890
export HTTPS_PROXY=http://192.168.99.128:7890
```

后续所有 `git clone`、`pip install`、`apt-get` 命令建议在同一 shell 会话里带上上述变量。

---

## 4. 安装 verl 0.5.x

### 4.1 安装 git（如缺失）

```bash
command -v git >/dev/null || apt-get update && apt-get install -y git
```

### 4.2 克隆 verl

```bash
git clone --branch v0.5.x --depth 1 https://github.com/volcengine/verl.git /opt/verl
```

> 仓库地址也可能是 `https://github.com/verl-project/verl.git`，以实际可访问的为准。

### 4.3 安装 verl 及依赖

```bash
cd /opt/verl

# 跳过 flash-attn（镜像已预装）和 pre-commit
grep -v "^flash-attn" requirements.txt | grep -v "^pre-commit" > /tmp/verl-requirements.txt
pip install -r /tmp/verl-requirements.txt

# 安装 verl（含 vllm extra，会拉 vllm 0.8.x）
pip install -e ".[vllm]"
```

### 4.4 安装后常见问题：torch 被降级

`pip install -e ".[vllm]"` 会把 **torch 从 2.7.1 降到 2.6.0**，导致 flash_attn 无法导入。

**处理思路：**

1. 先恢复镜像自带的 torch 2.7.1
2. 用 `--no-deps` 安装与 torch 2.7.1 匹配的 vllm 0.10.0
3. 手动补齐 vllm 运行时依赖

```bash
# 恢复 torch（如已被降级）
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 固定 transformers 在 4.x（避免 vllm / verl 冲突）
pip install "transformers>=4.51.1,<5.0" "tokenizers>=0.21,<0.22"

# 安装 vllm 0.10.0，不改动 torch
pip install vllm==0.10.0 --no-deps
```

---

## 5. 补齐 vllm 0.10.0 运行时依赖

`--no-deps` 不会自动安装 vllm 的传递依赖，需手动安装：

```bash
pip install \
  pybase64 cbor2 \
  "openai>=1.87.0,<=1.90.0" \
  compressed-tensors==0.10.2 \
  depyf==0.19.0 \
  "gguf>=0.13.0" \
  outlines_core==0.2.10 \
  xgrammar==0.1.21 \
  numba==0.61.2

# xformers 单独装，避免改 torch
pip install xformers==0.0.31 --no-deps
```

### 5.1 安装 C 编译器（triton JIT 需要）

vllm 首次初始化 GPU worker 时，triton 会 JIT 编译 CUDA 代码，容器里缺 gcc 会报错：

```bash
apt-get update && apt-get install -y build-essential
```

---

## 6. 准备 GSM8K 数据

在容器内执行 verl 自带预处理脚本：

```bash
cd /opt/verl
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

完成后应有：

```
~/data/gsm8k/train.parquet
~/data/gsm8k/test.parquet
```

训练脚本默认读取 `$HOME/data/gsm8k/*.parquet`。

---

## 7. 模型路径

容器内模型默认位于：

```
/opt/data/model_scope/Qwen/Qwen2.5-0.5B-Instruct
```

单卡训练可通过环境变量覆盖：

```bash
export MODEL_PATH=/opt/data/model_scope/Qwen/Qwen2.5-0.5B-Instruct
```

---

## 8. 验证环境

```bash
python3 -c "import torch; print('torch:', torch.__version__, torch.cuda.is_available())"
python3 -c "import flash_attn; print('flash_attn ok')"
python3 -c "import vllm; print('vllm:', vllm.__version__)"
python3 -c "import verl; print('verl:', getattr(verl, '__version__', 'ok'))"
python3 -c "import ray; print('ray:', ray.__version__)"
python3 -m verl.trainer.main_ppo --help | head -5
```

期望输出示例：

```
torch: 2.7.1+cu128 True
flash_attn ok
vllm: 0.10.0
verl: 0.5.0
ray: 2.47.1
```

---

## 9. 训练脚本相关修改

本仓库 `run_qwen2_5_0_5b_ppo_single_gpu.sh` 针对 verl 0.5.x 做了两处适配：

### 9.1 Critic FSDP 配置路径

verl 0.5.x Hydra 配置键为 `critic.model.fsdp_config.*`，不是 `critic.fsdp.*`：

```bash
critic.model.fsdp_config.param_offload=False
critic.model.fsdp_config.optimizer_offload=False
```

### 9.2 训练前自动启动 Ray

脚本会在调用 `main_ppo` 前执行：

```bash
ray stop --force 2>/dev/null || true
ray start --head \
    --dashboard-host=0.0.0.0 \
    --node-ip-address="$(hostname -I | awk '{print $1}')" \
    --port=6379
export RAY_ADDRESS=auto
```

避免 Ray 连接 `127.0.0.1:6379` 失败。

### 9.3 Reward 函数接口

verl 0.5.x / 0.6.x 实际调用方式为：

```python
compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
```

返回值需包含 `"score"` 字段。`gsm8k_reward.py` 已兼容该接口（课程文档中的 `reward_input` dict 写法与 verl 官方不一致）。

---

## 10. 启动训练

### 单卡

```bash
cd /root/hands-on-modern-rl/code/chapter08_rlhf/verl_gsm8k
chmod +x run_qwen2_5_0_5b_ppo_single_gpu.sh
./run_qwen2_5_0_5b_ppo_single_gpu.sh 2>&1 | tee log.txt
```

### 8 卡

```bash
./run_qwen2_5_0_5b_ppo_8gpu.sh 2>&1 | tee log8.txt
```

8 卡脚本会设置更大的 batch（`TRAIN_BATCH_SIZE=1024`、`ROLLOUT_N=4` 等），再调用单卡脚本。

### 指定 GPU

```bash
CUDA_VISIBLE_DEVICES=0 ./run_qwen2_5_0_5b_ppo_single_gpu.sh
```

---

## 11. 依赖版本冲突速查

| 现象 | 原因 | 处理 |
|------|------|------|
| `flash_attn` 导入失败 | torch 被 pip 降到 2.6.0 | 恢复 torch 2.7.1+cu128 |
| `vllm` 与 torch 2.7.1 ABI 冲突 | vllm 0.8.x 要求 torch 2.6.0 | 改用 vllm 0.10.0 `--no-deps` |
| `ModuleNotFoundError: cbor2` | vllm `--no-deps` 缺依赖 | 按第 5 节补装 |
| triton JIT 报 `gcc not found` | runtime 镜像无编译器 | `apt install build-essential` |
| `critic.fsdp` 配置报错 | verl 0.5.x 键名变更 | 改为 `critic.model.fsdp_config.*` |
| `compute_score() missing reward_input` | reward API 与 verl 不匹配 | 使用当前仓库里的 `gsm8k_reward.py` |
| Ray 连接失败 | 未启动 head 节点 | 脚本内 `ray start --head` 或手动启动 |

### vllm 与 torch 对应关系（实测）

| vllm | 要求 torch | 与镜像 torch 2.7.1 |
|------|-----------|-------------------|
| 0.8.3 ~ 0.8.5 | 2.6.0 | 不兼容 |
| 0.9.x | 2.7.0 | 接近但不精确 |
| **0.10.0** | **2.7.1** | **匹配** |
| 0.11.0+ | 2.8.0+ | 不兼容 |

---

## 12. 容器重建（环境丢失时）

若容器被删除，按顺序重建：

```bash
# 1. 宿主机启动容器
docker run -d \
  --name hands-on-modern-rl \
  --network host --gpus all --ipc=host \
  -v /root/hands-on-modern-rl:/root/hands-on-modern-rl \
  -v /opt/data:/opt/data \
  -w /root/hands-on-modern-rl \
  javirub/flashattention-pytorch:flashattn2.8.1-pytorch2.7.1-cuda12.8-cudnn9-runtime \
  sleep infinity

# 2. 进入容器，设置代理，依次执行第 4~6 节命令

# 3. 验证后启动训练
docker exec -it hands-on-modern-rl bash
```

> 容器内 pip 安装的包不会持久化到镜像；重建容器后需重新安装 verl / vllm 依赖。课程代码和 checkpoint 在挂载目录中，不会丢失。

---

## 13. 相关文件

| 路径 | 说明 |
|------|------|
| `run_qwen2_5_0_5b_ppo_single_gpu.sh` | 单卡 PPO 启动脚本 |
| `run_qwen2_5_0_5b_ppo_8gpu.sh` | 8 卡 PPO 启动脚本 |
| `gsm8k_reward.py` | GSM8K 规则 reward（verl 0.5.x 兼容） |
| `gsm8k_reward_advanced.py` | accuracy + format 组合 reward |
| `checkpoints/` | 训练 checkpoint 输出目录 |
| `/opt/verl` | 容器内 verl 源码（非挂载，重建容器需重装） |

---

*文档生成时间：2026-06-09，基于实际搭建与排错过程整理。*
