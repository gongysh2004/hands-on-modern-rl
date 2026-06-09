import re
from typing import Any

REWARD_NAME = "gsm8k"
REWARD_TYPE = "sequential"


def extract_answer(response: str) -> str | None:
    """从模型输出中提取最终答案。"""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return boxed[-1].strip()

    answer_tag = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_tag:
        return answer_tag.group(1).strip()

    lines = response.strip().split("\n")
    for line in reversed(lines):
        numbers = re.findall(r"-?\d+\.?\d*", line)
        if numbers:
            return numbers[-1]

    return None


def check_answer(predicted: str | None, ground_truth: str) -> float:
    """比较预测答案和标准答案。"""
    if predicted is None:
        return 0.0

    try:
        predicted_value = float(predicted.replace(",", ""))
        ground_truth_value = float(ground_truth.replace(",", ""))
        return 1.0 if abs(predicted_value - ground_truth_value) < 1e-6 else 0.0
    except (ValueError, TypeError):
        return 1.0 if predicted.strip() == ground_truth.strip() else 0.0


def compute_score(
    reward_input: dict[str, Any] | None = None,
    *,
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, float]:
    """veRL 自定义 reward 入口，兼容 verl 0.5.x 与新版 reward_input 接口。"""
    if reward_input is not None:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
    else:
        response = solution_str or ""
        if ground_truth is None:
            raise TypeError("compute_score() missing ground_truth")

    predicted = extract_answer(response)
    accuracy = check_answer(predicted, ground_truth)

    return {
        "score": accuracy,
        "overall": accuracy,
        "accuracy": accuracy,
        "format": 1.0 if predicted is not None else 0.0,
    }

