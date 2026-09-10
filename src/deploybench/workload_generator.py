"""Synthetic workload prompt generation optimized for fast generation and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING
from tqdm import tqdm

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


def _base_cache_prefix(workload: WorkloadSpec, hf_id: str, seed: int) -> str:
    """Computes a stable hash independent of num_prompts to allow superset reuse."""
    raw = f"{workload.id}:{hf_id}:{workload.prompt_tokens}:{workload.output_tokens}:{seed}:{workload.template}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_key(workload: WorkloadSpec, hf_id: str, seed: int, num_prompts: int | None = None) -> str:
    """Backward-compatible wrapper for cache key generation."""
    return _base_cache_prefix(workload, hf_id, seed)


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


def _generate_fast_filler(rng: random.Random, approx_words: int) -> str:
    """Fast random filler text generation without string concat overhead."""
    words = [rng.choice(FILLER_WORDS) for _ in range(approx_words)]
    return " ".join(words)

_generate_filler = _generate_fast_filler


def build_prompt_to_token_count(
    target_tokens: int,
    template: str,
    rng: random.Random,
    tokenizer,
    tolerance: float = 0.02,
) -> str:
    """
    Builds a prompt guaranteed to match target_tokens via single-pass tokenization and decoding.
    Replaces slow binary search loops with direct slicing.
    """
    prefix = TEMPLATE_PREFIXES.get(template, TEMPLATE_PREFIXES["chat"])
    if tokenizer is None:
        approx_chars = target_tokens * 4
        return (prefix + _generate_fast_filler(rng, target_tokens))[:approx_chars]

    # Generate slightly more words than needed (1 word is ~1.2 - 1.5 tokens)
    needed_words = int(target_tokens * 1.2) + 50
    raw_text = prefix + _generate_fast_filler(rng, needed_words)

    tokens = tokenizer.encode(raw_text, add_special_tokens=False)
    if len(tokens) >= target_tokens:
        tokens = tokens[:target_tokens]
    else:
        extra_tokens = tokenizer.encode(
            _generate_fast_filler(rng, target_tokens), add_special_tokens=False
        )
        tokens = (tokens + extra_tokens)[:target_tokens]

    return tokenizer.decode(tokens, skip_special_tokens=True)


# Backward-compatible alias
build_exact_token_prompt = build_prompt_to_token_count


def generate_needle_prompt(
    context_length: int,
    needle_position: float,
    trial: int,
    rng: random.Random,
    tokenizer,
) -> tuple[str, str, str]:
    """Generates a Needle-in-a-Haystack prompt for retrieval verification."""
    passphrase = f"BLUE-TIGER-{4000 + trial}"
    needle = f"The secret passphrase for run {trial} is: {passphrase}."
    question = f"What is the secret passphrase for run {trial}? Answer only the passphrase."

    filler_tokens = max(100, context_length - count_tokens(needle + question, tokenizer) - 20)
    filler = build_prompt_to_token_count(filler_tokens, "rag", rng, tokenizer)

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
    num_prompts: int | None = None,
) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    target_num_prompts = num_prompts if num_prompts is not None else getattr(workload, "num_prompts", 512)
    base_hash = _base_cache_prefix(workload, hf_id, seed)

    # 1. Reuse existing dataset if it already contains at least target_num_prompts
    candidate_files = sorted(
        GENERATED_DIR.glob(f"{workload.id}_{base_hash}_*.jsonl"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
        reverse=True,
    )

    for candidate in candidate_files:
        try:
            available_prompts = int(candidate.stem.split("_")[-1])
            if available_prompts >= target_num_prompts and not force_regenerate:
                logger.info(
                    "Reusing cached synthetic dataset (%d >= %d prompts) -> %s",
                    available_prompts, target_num_prompts, candidate,
                )
                return candidate
        except (ValueError, IndexError):
            continue

    out_path = GENERATED_DIR / f"{workload.id}_{base_hash}_{target_num_prompts}.jsonl"
    if out_path.exists() and not force_regenerate:
        logger.info("Found cached synthetic dataset (%d prompts) -> %s", target_num_prompts, out_path)
        return out_path

    logger.info(
        "Synthesizing %d prompts for '%s' (Prompt tokens: %d, Output tokens: %d)...",
        target_num_prompts,
        workload.id,
        workload.prompt_tokens,
        workload.output_tokens,
    )

    rng = random.Random(seed)
    tokenizer = _get_tokenizer(hf_id)
    template = getattr(workload, "template", None) or workload.type
    if template == "synthetic":
        template = "chat"

    records: list[dict] = []
    with tqdm(
        total=target_num_prompts,
        desc=f"Generating [{workload.id}]",
        unit="prompt",
        dynamic_ncols=True,
    ) as pbar:
        for i in range(target_num_prompts):
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
            pbar.update(1)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("Generated %d prompts -> %s", len(records), out_path)
    return out_path