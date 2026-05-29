# Synthetic Workloads

Generated JSONL datasets are written to `generated/` at benchmark runtime.

Each line contains:

```json
{
  "id": "sample_000001",
  "prompt": "...",
  "expected_output_tokens": 512,
  "metadata": {
    "workload_type": "rag",
    "target_prompt_tokens": 8192
  }
}
```

Prompt templates: `chat`, `coding`, `rag`, `turkish`, `needle`.
