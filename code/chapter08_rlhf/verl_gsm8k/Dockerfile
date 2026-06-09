# veRL 0.5.x + vLLM 0.10.0 + GSM8K PPO 训练环境
# 基于 docker-env-setup.md 固化，避免每次重建容器后手动装依赖
#
# 构建:
#   docker build -t hands-on-modern-rl:verl-gsm8k .
#
# 运行 (与文档一致，代码与模型目录挂载):
#   docker run -d \
#     --name hands-on-modern-rl \
#     --network host --gpus all --ipc=host \
#     -v /root/hands-on-modern-rl:/root/hands-on-modern-rl \
#     -v /opt/data:/opt/data \
#     -w /root/hands-on-modern-rl \
#     hands-on-modern-rl:verl-gsm8k

FROM javirub/flashattention-pytorch:flashattn2.8.1-pytorch2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HOME=/root

# 构建阶段如需代理，传入: --build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=...
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY}

# git: 克隆 verl；build-essential: vllm/triton JIT 编译需要 gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# verl 0.5.x
# ---------------------------------------------------------------------------
RUN git clone --branch v0.5.x --depth 1 https://github.com/volcengine/verl.git /opt/verl

WORKDIR /opt/verl

# 跳过 flash-attn（镜像已预装）和 pre-commit
RUN grep -v "^flash-attn" requirements.txt \
        | grep -v "^pre-commit" \
        > /tmp/verl-requirements.txt \
    && pip install --no-cache-dir -r /tmp/verl-requirements.txt

# 不装 [vllm] extra（setup.py 锁 vllm<=0.8.5 会降级 torch），只装 verl 本体
RUN pip install --no-cache-dir -e .

# 固定 transformers 4.x，直接装与 torch 2.7.1 匹配的 vllm 0.10.0
RUN pip install --no-cache-dir \
        "transformers>=4.51.1,<5.0" \
        "tokenizers>=0.21,<0.22" \
    && pip install --no-cache-dir vllm==0.10.0 --no-deps

# vllm --no-deps 缺失的运行时依赖
RUN pip install --no-cache-dir \
        pybase64 \
        cbor2 \
        "openai>=1.87.0,<=1.90.0" \
        compressed-tensors==0.10.2 \
        depyf==0.19.0 \
        "gguf>=0.13.0" \
        outlines_core==0.2.10 \
        xgrammar==0.1.21 \
        numba==0.61.2 \
    && pip install --no-cache-dir xformers==0.0.31 --no-deps

# ---------------------------------------------------------------------------
# GSM8K 数据（训练脚本默认读 $HOME/data/gsm8k/*.parquet）
# ---------------------------------------------------------------------------
RUN python3 examples/data_preprocess/gsm8k.py --local_dir /root/data/gsm8k

# ---------------------------------------------------------------------------
# 环境自检
# ---------------------------------------------------------------------------
RUN python3 -c "import torch; print('torch:', torch.__version__, torch.cuda.is_available())" \
    && python3 -c "import flash_attn; print('flash_attn ok')" \
    && python3 -c "import vllm; print('vllm:', vllm.__version__)" \
    && python3 -c "import verl; print('verl:', getattr(verl, '__version__', 'ok'))" \
    && python3 -c "import ray; print('ray:', ray.__version__)"

WORKDIR /root/hands-on-modern-rl

CMD ["sleep", "infinity"]
