# 12.4 动手：端到端 Agentic RL 训练——真实模型、真实 Benchmark、真实提升

前面的实验用模拟数据对比了 ORM 和 PRM。但那些轨迹是假的——没有真实的模型推理，没有真实的代码执行，也没有真实的梯度更新。这一节我们要做一件更硬核的事：**用真实的 Code LLM，构建真实的代码执行工具，模型自己写代码→执行→看报错→修复，用 rollout 出来的轨迹 finetune 模型，跑 HumanEval 看 pass@1 的真实提升。**

整个实验在一台 24GB 显存的 GPU（如 RTX 4090 / A5000）上即可完成。如果你没有 GPU，可以用 Google Colab 的免费 T4。

> 实验设计参考 SimpleTIR（ICLR 2026）、ReTool（字节跳动）和 Search-R1 的真实训练管线。它们的核心模式是一样的：模型生成工具调用 → 环境执行 → 结果追加到上下文 → 模型继续生成 → 收集完整轨迹 → 训练。

```mermaid
flowchart TD
    subgraph Rollout ["① Rollout：模型 + 工具环境交互"]
        M1["模型生成\n```python\ncode\n```"] --> E["工具：执行代码"]
        E --> O["观察：\n报错/通过"]
        O --> M2["模型继续\n修复代码"]
        M2 --> E
    end

    subgraph Collect ["② 收集轨迹"]
        T["完整轨迹:\n[prompt, response₁, obs₁, response₂, obs₂, ...]"]
    end

    subgraph Train ["③ Finetune"]
        S["路径 A:\nSFT on 成功轨迹"] 
        R["路径 B:\nGRPO RL\n（组采样 + 对比）"]
    end

    Rollout --> Collect --> Train

    style M1 fill:#e3f2fd,stroke:#1976d2,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style O fill:#fce4ec,stroke:#c62828,color:#000
    style M2 fill:#e3f2fd,stroke:#1976d2,color:#000
    style T fill:#f3e5f5,stroke:#7b1fa2,color:#000
    style S fill:#e8f5e9,stroke:#2e7d32,color:#000
    style R fill:#e8f5e9,stroke:#2e7d32,color:#000
```

## 第零步：环境准备

```bash
pip install torch transformers accelerate datasets
pip install matplotlib numpy peft
```

我们使用 **Qwen2.5-Coder-1.5B-Instruct** 作为基座模型。1.5B 参数量在 24GB 显存上跑 GRPO（group_size=4）绰绰有余。

```python
# ==========================================
# 0. 全局配置
# ==========================================
import torch, numpy as np, random, re, os, subprocess, tempfile, warnings
warnings.filterwarnings("ignore")

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 512
GROUP_SIZE = 4
MAX_EPOCHS = 3
LR = 5e-6
KL_COEFF = 0.05

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device: {device}")
```

## 第一步：加载模型 + 代码执行沙箱 + HumanEval

```python
# ==========================================
# 1.1 加载模型
# ==========================================
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"Model: {MODEL_NAME} ({sum(p.numel() for p in model.parameters())/1e9:.2f}B)")

# ==========================================
# 1.2 加载 HumanEval
# ==========================================
humaneval = load_dataset("openai_humaneval", trust_remote_code=True)
problems = humaneval["test"]
print(f"HumanEval: {len(problems)} problems")

# ==========================================
# 1.3 代码执行沙箱（和 SimpleTIR 一样：写文件 → subprocess → 拿结果）
# ==========================================
def sandbox_execute(code: str, task_prompt: str, task_test: str,
                    entry_point: str, timeout: float = 10.0) -> dict:
    """
    在子进程中真实执行代码，返回 stdout/stderr。
    这就是 SimpleTIR 的 Sandbox Fusion 做的事——只不过我们用本地 subprocess。
    """
    full_code = task_prompt + code + "\n" + task_test + "\n"
    full_code += f"\ncheck({entry_point})\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "solution.py")
        with open(path, "w") as f:
            f.write(full_code)
        try:
            r = subprocess.run(["python", path], capture_output=True, text=True,
                               timeout=timeout, cwd=tmpdir)
            passed = r.returncode == 0
            error = None if passed else (r.stderr.strip().split("\n")[-1] if r.stderr else "unknown")
            return {"passed": passed, "error": error, "stdout": r.stdout.strip()[:200]}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "TIMEOUT", "stdout": ""}


def batch_evaluate(tasks, completions):
    results = [sandbox_execute(c, t["prompt"], t["test"], t["entry_point"]) for t, c in zip(tasks, completions)]
    passed = sum(1 for r in results if r["passed"])
    return {"pass@1": passed / len(results), "passed": passed, "total": len(results), "details": results}
```

## 第二步：基线评测——训练前模型的真实水平

先跑一遍 HumanEval（单轮补全，不调工具），记录基线。

```python
# ==========================================
# 2. 基线评测（单轮补全，无工具调用）
# ==========================================
BASELINE_N = 64
baseline_tasks = list(problems.select(range(BASELINE_N)))

def generate_single_turn(task, temperature=0.0):
    messages = [{"role": "user", "content": f"Complete this function. Output ONLY the function body:\n\n{task['prompt']}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             temperature=temperature, do_sample=temperature>0,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print(f"Baseline evaluation on {BASELINE_N} HumanEval problems...")
baseline_completions = []
for i, task in enumerate(baseline_tasks):
    comp = generate_single_turn(task)
    res = sandbox_execute(comp, task["prompt"], task["test"], task["entry_point"])
    baseline_completions.append(comp)
    if (i+1) % 16 == 0:
        so_far = sum(1 for c in baseline_completions[:i+1]
                     if sandbox_execute(c, baseline_tasks[baseline_completions.index(c)]["prompt"],
                                       baseline_tasks[baseline_completions.index(c)]["test"],
                                       baseline_tasks[baseline_completions.index(c)]["entry_point"])["passed"])

# 更高效的基线计算
baseline_metrics = batch_evaluate(baseline_tasks, baseline_completions)
print(f"Baseline pass@1: {baseline_metrics['pass@1']:.1%} ({baseline_metrics['passed']}/{baseline_metrics['total']})")
```

## 第三步：构建 Agent Loop——模型生成代码，沙箱执行，观察结果，继续生成

这是核心。参考 SimpleTIR 的做法：模型在推理过程中生成 ```` ```python ``` ````  代码块，沙箱执行后返回结果（stdout 或报错），结果追加到上下文，模型继续生成。**整条轨迹就是训练数据。**

### 3.1 Agent Prompt

```python
# ==========================================
# 3.1 Agent System Prompt
#     参考 SimpleTIR：模型自由生成代码块，不预设工具调用格式
# ==========================================
AGENT_PROMPT = """\
You are a Python coding agent. Given a function signature, implement it.

You can write Python code blocks to test your implementation. Format:
```python
# your code here
```
The code will be executed and you'll see the result. If tests fail, fix the code and try again.

When confident, output your final implementation in a code block. You have at most 3 execution rounds."""
```

### 3.2 Agent Rollout——真实的多轮交互

参考 Search-R1 的 `LLMGenerationManager.run_llm_loop()` 和 ReTool 的 `CustomSandboxFusionTool`。

**Search-R1 的做法**：维护一个 rolling state，每轮模型生成到 `</search>` 截断，搜索引擎返回结果后追加到 rolling state，下一轮模型看到完整历史继续生成。同时维护两个平行张量：`responses`（真实 token）和 `responses_with_info_mask`（工具 token 替换为 pad_id）。

**ReTool 的做法**：模型生成 ```` ```python ``` ```` 代码块，正则 `r"```python(.*?)```"` 提取代码，沙箱执行后 stdout/stderr 作为 `tool` role 消息注入。如果最后一行没有 `print()`，自动补上。

```python
# ==========================================
# 3.2 Agent Rollout
#     参考 Search-R1 generation.py + ReTool retool.py
# ==========================================

# ★ 和 ReTool 完全一样的代码块提取正则
CODE_PATTERN = re.compile(r"```(?:python|py)?\n(.*?)\n```", re.DOTALL)

def ensure_print(code: str) -> str:
    """
    ReTool 的做法：如果最后一行不是 print()，自动补上。
    这样沙箱执行时最后一行的结果会被输出到 stdout。
    """
    lines = code.strip().split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() and not lines[i].strip().startswith("print"):
            lines[i] = f"print({lines[i]})"
            break
    return "\n".join(lines)

def run_agent_rollout(task, temperature=0.7, max_turns=3, verbose=False):
    """
    多轮 Agent Rollout。
    流程参考 Search-R1 的 run_llm_loop + ReTool 的 code_interpreter：

    1. 模型生成（可能包含 ```python``` 代码块）
    2. 提取代码块（ReTool 的 CODE_PATTERN 正则）
    3. 沙箱执行（ReTool 的 SandboxFusion / 我们的 subprocess）
    4. 执行结果作为 observation 追加（Search-R1 的 next_obs）
    5. 下一轮模型看到完整历史（Search-R1 的 rolling state）
    6. 收集完整轨迹（用于后续 finetune）
    """
    conversation = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": f"Implement this function:\n\n{task['prompt']}"},
    ]

    current_code = ""
    passed = False
    is_valid = True  # ★ SimpleTIR 的 void turn 检测：这轮是否产出了有效代码

    for turn_idx in range(max_turns):
        # --- Step 1: 模型生成（Search-R1 的 generate_sequences） ---
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=temperature, do_sample=temperature>0,
                                 top_p=0.95, pad_token_id=tokenizer.pad_token_id)

        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # ★ Search-R1 的 postprocess：追加 assistant 回复到轨迹
        conversation.append({"role": "assistant", "content": response})

        # --- Step 2: 提取代码块（ReTool 的 CODE_PATTERN） ---
        code_blocks = CODE_PATTERN.findall(response)

        if not code_blocks:
            # ★ SimpleTIR 的 void turn：这轮没有产出有效代码块
            is_valid = False
            current_code = response
            break

        # ★ ReTool 的做法：取最后一个代码块，确保有 print()
        current_code = ensure_print(code_blocks[-1].strip())

        # --- Step 3: 沙箱执行（ReTool 的 SandboxFusion / Search-R1 的 search） ---
        exec_result = sandbox_execute(current_code, task["prompt"], task["test"], task["entry_point"])

        if verbose:
            status = "PASS" if exec_result["passed"] else f"FAIL: {exec_result['error'][:50]}"
            print(f"  Turn {turn_idx+1}: {status}")

        # --- Step 4: 执行结果作为 observation（Search-R1 的 next_obs） ---
        if exec_result["passed"]:
            # ★ Search-R1 的 info 追加方式
            conversation.append({"role": "user",
                "content": f"<output>\nALL TESTS PASSED\n</output>\nOutput your final implementation."})
            passed = True
            # 再让模型生成最终代码
            text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     temperature=temperature, do_sample=temperature>0,
                                     pad_token_id=tokenizer.pad_token_id)
            final_resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            conversation.append({"role": "assistant", "content": final_resp})
            final_blocks = CODE_PATTERN.findall(final_resp)
            if final_blocks:
                current_code = final_blocks[-1].strip()
            break
        else:
            # ★ Search-R1：报错信息作为 observation
            conversation.append({"role": "user",
                "content": f"<output>\nFAILED: {exec_result['error']}\n</output>\nFix the code and try again."})

    # 最终验证
    if current_code:
        final_check = sandbox_execute(current_code, task["prompt"], task["test"], task["entry_point"])
        passed = final_check["passed"]
    else:
        passed = False

    return {
        "conversation": conversation,   # ★ 完整轨迹——训练用
        "completion": current_code,      # 最终代码——评测用
        "passed": passed,                # 是否通过——reward 用
        "is_valid": is_valid,            # ★ SimpleTIR 的 void turn 过滤标记
        "turns": len([m for m in conversation if m["role"] == "assistant"]),
    }
```

### 3.3 验证 Rollout

```python
# 快速验证 rollout 是否正常工作
print("Sanity check — Agent Rollout:")
print("-" * 60)
for i in [0, 5, 10]:
    task = problems[i]
    result = run_agent_rollout(task, temperature=0.3, verbose=True)
    print(f"  {task['task_id']}: {'PASS' if result['passed'] else 'FAIL'} "
          f"(turns: {result['turns']}, conversation: {len(result['conversation'])} messages)")
print("-" * 60)
```

## 第四步：收集 Rollout 轨迹 → 构建 Finetune 数据

Rollout 出来的轨迹是原始对话。要变成 finetune 数据，需要 tokenize 并构建 **info_mask**——Search-R1 证明了这个 mask 对训练效果至关重要（有 mask: 0.431 vs 无 mask: 0.343）。

### Search-R1 的 info_mask 机制

Search-R1 维护**两个平行张量**：
- `responses`：真实的 token 序列（包含 LLM 生成的 + 工具返回的）
- `responses_with_info_mask`：工具返回的 token 被替换为 `pad_token_id`

最终 `info_mask = create_attention_mask(responses_with_info_mask)`，pad → 0, 其余 → 1。这个 mask 同时用于 loss 和 KL 计算。

```python
# ==========================================
# 4. Tokenize 轨迹 + 构建 info_mask
#    参考 Search-R1 的 _info_masked_concatenate_with_padding
#    和 veRL 的 response_mask 机制
# ==========================================

def tokenize_with_info_mask(conversation, max_length=2048):
    """
    将多轮对话 tokenize，构建两个输出：
    - input_ids: 完整 token 序列（包含工具结果）
    - info_mask: 1=LLM 生成的 token, 0=工具返回的 token

    ★ 这就是 Search-R1 的 responses + responses_with_info_mask 模式。
    """
    pad_id = tokenizer.pad_token_id

    # 逐段 tokenize，记录哪些 token 是 assistant 生成的
    all_tokens = []          # 真实 token（用于 input_ids）
    all_tokens_masked = []   # mask 版本（工具 token 替换为 pad_id，用于 info_mask）
    prev_text = ""

    for msg in conversation:
        is_assistant = (msg["role"] == "assistant")

        partial = conversation[:conversation.index(msg) + 1]
        if is_assistant:
            full_text = tokenizer.apply_chat_template(partial, tokenize=False, add_generation_prompt=False)
        else:
            full_text = tokenizer.apply_chat_template(partial, tokenize=False, add_generation_prompt=True)

        new_text = full_text[len(prev_text):] if len(full_text) > len(prev_text) else ""
        if new_text:
            new_tokens = tokenizer.encode(new_text, add_special_tokens=False)
            all_tokens.extend(new_tokens)
            # ★ Search-R1 的核心：assistant token 保留原 id，其余替换为 pad_id
            if is_assistant:
                all_tokens_masked.extend(new_tokens)   # LLM 生成的 → 保留
            else:
                all_tokens_masked.extend([pad_id] * len(new_tokens))  # 工具/系统 → pad_id
        prev_text = full_text

    # 截断
    if len(all_tokens) > max_length:
        all_tokens = all_tokens[:max_length]
        all_tokens_masked = all_tokens_masked[:max_length]

    input_ids = torch.tensor([all_tokens], dtype=torch.long)
    # ★ info_mask: 从 masked 版本创建（和 Search-R1 的 create_attention_mask 一样）
    masked_ids = torch.tensor([all_tokens_masked], dtype=torch.long)
    info_mask = (masked_ids != pad_id).long()  # 非 pad → 1, pad → 0

    # labels 也构建：只有 assistant token 参与 loss
    labels = input_ids.clone()
    labels[info_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "info_mask": info_mask,  # ★ Search-R1 的 info_mask：LLM token=1, tool token=0
        "num_assistant_tokens": info_mask.sum().item(),
    }
```

### 4.1 批量 Rollout 收集轨迹

```python
# ==========================================
# 4.1 批量 Rollout：对训练集的每道题跑 agent loop
# ==========================================
TRAIN_IDS = list(range(64, 96))  # HumanEval #64-#95（避开评测集）
train_tasks = list(problems.select(TRAIN_IDS))

print(f"Collecting agent rollouts on {len(train_tasks)} tasks...")
print(f"Each task: {GROUP_SIZE} trajectories (group_size for GRPO)")
print("=" * 60)

all_trajectories = []  # 存储所有轨迹

for task_idx, task in enumerate(train_tasks):
    for g in range(GROUP_SIZE):
        result = run_agent_rollout(task, temperature=0.7, max_turns=3)
        result["task"] = task
        all_trajectories.append(result)

    if (task_idx + 1) % 8 == 0:
        passed = sum(1 for t in all_trajectories if t["passed"])
        print(f"  Task {task_idx+1}/{len(train_tasks)} | "
              f"Rollouts: {len(all_trajectories)} | "
              f"Passed: {passed} ({passed/len(all_trajectories):.1%})")

print(f"\nTotal trajectories: {len(all_trajectories)}")
print(f"Passed: {sum(1 for t in all_trajectories if t['passed'])}")
print(f"Avg turns: {np.mean([t['turns'] for t in all_trajectories]):.1f}")
```

## 第五步：两条 Finetune 路径

收集到轨迹后，有两条路可以走：

- **路径 A：Rejection Sampling + SFT**——只保留成功轨迹，做监督微调。最简单，效果稳定。
- **路径 B：GRPO RL**——组内比较，策略梯度更新。更高级，能从失败轨迹中学到东西。

### 路径 A：SFT on 成功轨迹（参考 ReTool 的冷启动阶段）

```python
# ==========================================
# 5A. SFT Finetune：只保留成功的 rollout 轨迹
#     参考 ReTool 第一阶段 + Search-R1 的拒绝采样
# ==========================================
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW

# 过滤：只保留通过测试的轨迹
success_trajs = [t for t in all_trajectories if t["passed"]]
print(f"成功轨迹: {len(success_trajs)}/{len(all_trajectories)} "
      f"({len(success_trajs)/len(all_trajectories):.1%})")

if len(success_trajs) == 0:
    print("没有成功轨迹！需要调整模型或温度。跳过 SFT。")
else:
    # ★ SimpleTIR 的 void turn 过滤：丢弃 is_valid=False 的轨迹
    valid_trajs = [t for t in success_trajs if t.get("is_valid", True)]
    print(f"Void turn 过滤: {len(success_trajs)} → {len(valid_trajs)} (丢弃 {len(success_trajs)-len(valid_trajs)} 条无效轨迹)")
    if len(valid_trajs) == 0:
        valid_trajs = success_trajs  # fallback
else:
    # 设置 LoRA
    model.enable_input_require_grads()
    model_sft = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
    ))
    model_sft.train()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model_sft.parameters()), lr=LR)

    print(f"\nSFT Training on {len(valid_trajs)} successful trajectories...")
    print(f"Trainable params: {sum(p.numel() for p in model_sft.parameters() if p.requires_grad)/1e6:.1f}M")

    for epoch in range(MAX_EPOCHS):
        random.shuffle(valid_trajs)
        total_loss = 0

        for traj in valid_trajs:
            enc = tokenize_with_info_mask(traj["conversation"])
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = enc["labels"].to(device)

            if (labels != -100).sum() == 0:
                continue  # 跳过没有 assistant token 的轨迹

            outputs = model_sft(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_sft.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(valid_trajs)
        print(f"  SFT Epoch {epoch+1}/{MAX_EPOCHS} | Avg Loss: {avg_loss:.4f}")

    model_sft.eval()
    print("SFT training complete.")
```

### 路径 B：GRPO RL（参考 SimpleTIR / DeepSWE）

```python
# ==========================================
# 5B. GRPO RL Finetune：组采样 → 组内比较 → 策略梯度
#     参考 SimpleTIR 的 GRPO+DAPO 和 Search-R1 的 token masking
# ==========================================

# 重新加载模型（SFT 和 RL 用不同的模型实例）
model_rl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
model_rl.enable_input_require_grads()
model_rl = get_peft_model(model_rl, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
    lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
))

# Reference model（KL 约束）
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer_rl = AdamW(filter(lambda p: p.requires_grad, model_rl.parameters()), lr=LR)

training_log = {"epoch": [], "success_rate": [], "mean_reward": [], "loss": []}

print("=" * 60)
print("GRPO RL Training (Multi-Turn Agent Rollout)")
print("=" * 60)

for epoch in range(MAX_EPOCHS):
    model_rl.train()
    epoch_rewards, epoch_losses = [], []
    epoch_successes = 0
    random.shuffle(train_tasks)

    for task_idx, task in enumerate(train_tasks):
        # ---- Phase 1: On-Policy Rollout (GROUP_SIZE 条轨迹) ----
        trajectories = []
        for g in range(GROUP_SIZE):
            result = run_agent_rollout(task, temperature=0.7, max_turns=3)
            result["task"] = task
            # Reward 设计（参考 ReTool retool.py + Search-R1 qa_em.py）：
            # 1. Outcome reward：二元 0/1
            base_reward = 1.0 if result["passed"] else 0.0
            # 2. ★ ReTool 的 tool-call shaping：答错时鼓励多调工具
            if base_reward == 0 and result["turns"] > 1:
                tool_bonus = (result["turns"] - 2) / 2 * 0.1
                base_reward = max(-0.6, tool_bonus)
            result["reward"] = base_reward
            trajectories.append(result)

        # ---- Phase 2: GRPO Advantage ----
        rewards = np.array([t["reward"] for t in trajectories])
        mean_r, std_r = rewards.mean(), rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # ---- Phase 3: 策略梯度更新（在完整轨迹上） ----
        for traj, advantage in zip(trajectories, advantages):
            if not traj["completion"]:
                continue

            enc = tokenize_with_info_mask(traj["conversation"])
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = enc["labels"].to(device)

            if (labels != -100).sum() == 0:
                continue

            # Policy log prob（只在 assistant token 上）
            outputs = model_rl(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            policy_lp = -outputs.loss

            # Reference log prob（KL）
            with torch.no_grad():
                ref_out = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ref_lp = -ref_out.loss

            # GRPO loss
            kl = policy_lp - ref_lp
            loss = -advantage * policy_lp + KL_COEFF * kl

            optimizer_rl.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_rl.parameters(), 1.0)
            optimizer_rl.step()
            epoch_losses.append(loss.item())

        epoch_rewards.extend(rewards.tolist())
        epoch_successes += sum(1 for t in trajectories if t["passed"])

        if (task_idx + 1) % 8 == 0:
            sr = epoch_successes / ((task_idx+1) * GROUP_SIZE)
            print(f"  Epoch {epoch+1} | Task {task_idx+1}/{len(train_tasks)} | SR: {sr:.1%}")

    # Epoch summary
    sr = epoch_successes / (len(train_tasks) * GROUP_SIZE)
    training_log["epoch"].append(epoch+1)
    training_log["success_rate"].append(sr)
    training_log["mean_reward"].append(np.mean(epoch_rewards))
    training_log["loss"].append(np.mean(epoch_losses) if epoch_losses else 0)
    print(f"  Epoch {epoch+1} Summary: SR={sr:.1%}, Reward={np.mean(epoch_rewards):.3f}, "
          f"Loss={training_log['loss'][-1]:.4f}")

model_rl.eval()
```

## 第六步：Benchmark 评测——真的提升了吗？

**不管训练过程多么花哨，只有独立 Benchmark 上的 pass@1 才是唯一裁判。**

```python
# ==========================================
# 6. 训练后 Benchmark 评测
# ==========================================

def evaluate_model(model_to_eval, tasks, label="Model"):
    """用 greedy decoding 评测模型在 HumanEval 上的 pass@1"""
    model_to_eval.eval()
    completions = []
    for i, task in enumerate(tasks):
        messages = [{"role": "user", "content": f"Complete this function. Output ONLY the function body:\n\n{task['prompt']}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_to_eval.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                          temperature=0.0, pad_token_id=tokenizer.pad_token_id)
        comp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        completions.append(comp)
        if (i+1) % 16 == 0:
            passed_so_far = sum(1 for c, t in zip(completions, tasks[:i+1])
                               if sandbox_execute(c, t["prompt"], t["test"], t["entry_point"])["passed"])
            print(f"  [{label}] {i+1}/{len(tasks)} running pass@1: {passed_so_far/(i+1):.1%}")

    metrics = batch_evaluate(tasks, completions)
    print(f"  [{label}] Final pass@1: {metrics['pass@1']:.1%} ({metrics['passed']}/{metrics['total']})")
    return metrics

print("=" * 60)
print("POST-TRAINING Evaluation")
print("=" * 60)

# 评测 SFT 模型
if len(success_trajs) > 0:
    sft_metrics = evaluate_model(model_sft, baseline_tasks, "SFT")
else:
    sft_metrics = baseline_metrics

# 评测 GRPO 模型
rl_metrics = evaluate_model(model_rl, baseline_tasks, "GRPO-RL")

# 对比
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"  Baseline (no training):  {baseline_metrics['pass@1']:.1%}")
print(f"  SFT (on success trajs):  {sft_metrics['pass@1']:.1%}  (Δ = {sft_metrics['pass@1'] - baseline_metrics['pass@1']:+.1%})")
print(f"  GRPO RL (on all trajs):  {rl_metrics['pass@1']:.1%}  (Δ = {rl_metrics['pass@1'] - baseline_metrics['pass@1']:+.1%})")
print("=" * 60)
```

### 可视化

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左图：Benchmark 对比 ---
ax = axes[0]
methods = ['Baseline', 'SFT\n(success only)', 'GRPO RL\n(all trajs)']
pass_rates = [baseline_metrics["pass@1"], sft_metrics["pass@1"], rl_metrics["pass@1"]]
colors = ['#90a4ae', '#42a5f5', '#66bb6a']

bars = ax.bar(methods, pass_rates, color=colors, edgecolor=[c.replace('a', '')[:7] for c in colors], linewidth=2)
ax.set_ylabel('pass@1')
ax.set_title(f'HumanEval Benchmark (n={BASELINE_N})', fontweight='bold')
ax.set_ylim(0, max(pass_rates) * 1.3)

for bar, v in zip(bars, pass_rates):
    ax.text(bar.get_x() + bar.get_width()/2., v + 0.015,
            f'{v:.1%}', ha='center', fontsize=13, fontweight='bold')

# 标注最佳
best = np.argmax(pass_rates)
if pass_rates[best] > pass_rates[0]:
    ax.annotate(f'Best: +{pass_rates[best]-pass_rates[0]:.1%}',
                xy=(best, pass_rates[best]), xytext=(best, pass_rates[best]+0.08),
                fontsize=13, fontweight='bold', color='#2e7d32',
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))

# --- 右图：GRPO 训练过程 ---
ax = axes[1]
epochs = training_log["epoch"]
ax.plot(epochs, training_log["success_rate"], 'o-', color='#388e3c', lw=2, label='Agent Success Rate')
ax.plot(epochs, training_log["mean_reward"], 's--', color='#1976d2', lw=2, label='Mean Reward')
ax.set_xlabel('Epoch')
ax.set_title('GRPO RL Training Progress', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Agentic RL: Real Model + Real Rollout + Real Benchmark', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("agentic_rl_real_benchmark.png", dpi=150)
print("Saved: agentic_rl_real_benchmark.png")
```

## 参考：真正在做 Agentic RL 训练的开源项目

| 项目 | 工具环境 | Rollout 方式 | 训练方法 | 结果 | 代码 |
|------|---------|-------------|---------|------|------|
| **SimpleTIR** | 代码沙箱 | 模型写 ````python```` → 执行 → 看结果 | GRPO + void turn 过滤 | AIME 22→50, ICLR'26 | [GitHub](https://github.com/ltzheng/SimpleTIR) |
| **ReTool** | 代码解释器 | 推理中穿插代码块执行 | SFT 冷启动 → RL | AIME 67% | [GitHub](https://github.com/ReTool-RL/ReTool) |
| **Search-R1** | 搜索引擎 | 推理→`<search>`→搜索结果→继续 | PPO/GRPO + token mask | 多轮搜索RL标杆 | [GitHub](https://github.com/PeterGriffinJin/Search-R1) |
| **DeepSWE** | Docker + 测试套件 | 多步 agent 修 bug | GRPO++ | SWE-bench 42% | [GitHub](https://github.com/agentica-project/rllm) |
| **VerlTool** | 通用工具服务器 | 异步多轮交互 | GRPO | 多域 SOTA | [GitHub](https://github.com/TIGER-AI-Lab/verl-tool) |

## 实验总结

**核心管线**（所有项目共同遵循）：

1. **构建工具环境**（代码沙箱 / 搜索引擎 / Docker）
2. **模型与环境多轮 Rollout**：模型生成工具调用 → 环境执行 → 结果追加到上下文 → 模型继续生成
3. **收集完整轨迹**：`[prompt, assistant₁, observation₁, assistant₂, observation₂, ...]`
4. **Finetune**：
   - **路径 A（SFT）**：只保留成功轨迹，做监督微调。简单稳定。
   - **路径 B（GRPO RL）**：组采样 + 组内比较 + 策略梯度。能从失败中学习。
5. **独立 Benchmark 评测**：只有 pass@1 的变化才是真实的训练效果

**关键设计决策**：

- **Token mask**：工具执行结果的 token 不参与 loss（`labels = -100`），只有 assistant token 参与。这是 Search-R1 论文中实验证明有效的（有 mask: 0.431 vs 无 mask: 0.343）
- **Outcome reward**：只看最终测试是否通过（`passed = 0/1`），不需要每步给 reward。SimpleTIR 和 DeepSWE 都证实 outcome reward 足够
- **Code block 格式**：用 ```` ```python ``` ```` 而不是自定义 XML 标签。这是 SimpleTIR 和 ReTool 共同采用的模式，模型更容易学会

下一节我们聚焦 Agentic RL 的工程挑战——[怎么把这些想法变成一个真正能跑的训练系统](./agentic-engineering)。
