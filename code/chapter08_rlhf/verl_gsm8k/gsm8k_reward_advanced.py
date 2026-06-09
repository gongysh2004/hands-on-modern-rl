import re
from typing import Any

REWARD_NAME = "gsm8k_advanced"
REWARD_TYPE = "sequential"


def format_reward(response: str) -> float:
    """检查回答是否包含规范的推理过程。"""
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    has_reasoning = len(lines) >= 2
    has_answer_marker = bool(re.search(r"####|\\boxed|<answer>", response))

    score = 0.0
    if has_reasoning:
        score += 0.3
    if has_answer_marker:
        score += 0.2
    return score


def accuracy_reward(response: str, ground_truth: str) -> float:
    """检查最终答案是否正确。"""
    answer_match = re.search(r"####\s*(.+)", response)
    if answer_match:
        predicted = answer_match.group(1).strip()
    else:
        numbers = re.findall(r"-?\d+\.?\d*", response)
        predicted = numbers[-1] if numbers else None

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
    """veRL 自定义 reward 入口：accuracy 占 75%，format 占 25%。"""
    if reward_input is not None:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
    else:
        response = solution_str or ""
        if ground_truth is None:
            raise TypeError("compute_score() missing ground_truth")

    accuracy = accuracy_reward(response, ground_truth)
    format_score = format_reward(response)
    overall = 0.75 * accuracy + 0.25 * format_score

    return {
        "score": overall,
        "overall": overall,
        "accuracy": accuracy,
        "format": format_score,
    }

