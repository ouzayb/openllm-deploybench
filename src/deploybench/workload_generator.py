"""Synthetic workload prompt generation."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

from deploybench.config import WorkloadSpec
from deploybench.utils import PROJECT_ROOT

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

GENERATED_DIR = PROJECT_ROOT / "workloads" / "synthetic" / "generated"

TEMPLATE_PREFIXES = {
    "chat": "User: Please discuss the following topic in detail.\nTopic: ",
    "coding": "# Task: Implement the following specification.\n# Specification:\n",
    "rag": "Document section:\n",
    "turkish": "Kullanıcı: Aşağıdaki konuyu ayrıntılı olarak açıklayın.\nKonu: ",
    "needle": "",
}

FILLER_WORDS = [
    "analysis", "benchmark", "deployment", "inference", "latency", "throughput",
    "memory", "tensor", "parallel", "context", "token", "model", "server",
    "hardware", "performance", "optimization", "pipeline", "workload",
]


def _cache_key(workload: WorkloadSpec, hf_id: str, seed: int) -> str:
    raw = f"{workload.id}:{hf_id}:{workload.prompt_tokens}:{workload.output_tokens}:{workload.num_prompts}:{seed}:{workload.template}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_tokenizer(hf_id: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    except Exception as e:
        logger.warning("Tokenizer load failed for %s: %s; using char heuristic", hf_id, e)
        return None


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return max(1, len(text) // 4)
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return max(1, len(text) // 4)


def _generate_filler(rng: random.Random, words_needed: int) -> str:
    parts: list[str] = []
    while len(" ".join(parts).split()) < words_needed:
        parts.append(rng.choice(FILLER_WORDS))
        if rng.random() < 0.3:
            parts.append(rng.choice(FILLER_WORDS) + ".")
    return " ".join(parts)


def build_prompt_to_token_count(
    target_tokens: int,
    template: str,
    rng: random.Random,
    tokenizer,
    tolerance: float = 0.02,
) -> str:
    prefix = TEMPLATE_PREFIXES.get(template, TEMPLATE_PREFIXES["chat"])
    low = max(1, int(target_tokens * (1 - tolerance)))
    high = int(target_tokens * (1 + tolerance))

    # Binary search on word count
    word_lo, word_hi = 50, target_tokens * 8
    best = prefix + _generate_filler(rng, target_tokens * 2)
    for _ in range(32):
        mid = (word_lo + word_hi) // 2
        text = prefix + _generate_filler(rng, mid)
        n = count_tokens(text, tokenizer)
        if low <= n <= high:
            return text
        if n < low:
            word_lo = mid + 1
        else:
            word_hi = mid - 1
        best = text
    return best


def generate_needle_prompt(
    context_length: int,
    needle_position: float,
    trial: int,
    rng: random.Random,
    tokenizer,
) -> tuple[str, str, str]:
    passphrase = f"BLUE-TIGER-{4000 + trial}"
    needle = f"The secret passphrase for run {trial} is: {passphrase}."
    question = f"What is the secret passphrase for run {trial}? Answer only the passphrase."

    filler_tokens = max(100, context_length - count_tokens(needle + question, tokenizer) - 20)
    filler = build_prompt_to_token_count(filler_tokens, "rag", rng, tokenizer)

    # Insert needle at approximate position
    words = filler.split()
    insert_at = int(len(words) * needle_position)
    words.insert(insert_at, needle)
    body = " ".join(words)
    prompt = body + "\n\n" + question
    return prompt, passphrase, question


def generate_synthetic_dataset(
    workload: WorkloadSpec,
    hf_id: str,
    seed: int = 42,
    force_regenerate: bool = False,
) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(workload, hf_id, seed)
    out_path = GENERATED_DIR / f"{workload.id}_{key}.jsonl"
    if out_path.exists() and not force_regenerate:
        return out_path

    rng = random.Random(seed)
    tokenizer = _get_tokenizer(hf_id)
    template = getattr(workload, "template", None) or workload.type
    if template == "synthetic":
        template = "chat"

    records: list[dict] = []
    for i in range(workload.num_prompts):
        prompt = build_prompt_to_token_count(
            workload.prompt_tokens,
            template,
            rng,
            tokenizer,
        )
        records.append(
            {
                "id": f"sample_{i:06d}",
                "prompt": prompt,
                "expected_output_tokens": workload.output_tokens,
                "metadata": {
                    "workload_type": template,
                    "target_prompt_tokens": workload.prompt_tokens,
                },
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("Generated %d prompts -> %s", len(records), out_path)
    return out_path
