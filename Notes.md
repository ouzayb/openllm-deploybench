# Experimental Setup & Configuration Notes

## Dual NVIDIA RTX 5090 Evaluation Framework

### Qwen 2.5 Model Suite Benchmark Configuration

* **Hardware & Topology**
  * **System:** Dual NVIDIA GeForce RTX 5090 (`owned_rtx5090_2x_01`, 32 GB VRAM per card, PCIe Gen 5).
  * **Tensor Parallelism:** `tensor_parallel_size = 2` uniformly applied across all evaluated models to distribute weights evenly across both GPUs.

* **Model Suite & Quantization**
  * **Evaluated Models:** Qwen 2.5 family including **7B, 14B, 32B (AWQ), and 72B (AWQ)** variants.
  * **Quantization:** AWQ (`dtype=auto`) utilized for compressed variants to fit effectively within hardware budgets.

* **vLLM Engine & Runtime Parameters**
  * **Engine Version:** vLLM v1 engine enabled (`VLLM_USE_V1=1`).
  * **Sampler Backend:** FlashInfer sampler explicitly disabled (`VLLM_USE_FLASHINFER_SAMPLER=0`) globally for portability and stability.
  * **Optimizations:** `enable_chunked_prefill` and `enable_prefix_caching` activated across all benchmark runs.
  * **Warmup Strategy:** Dynamically scaled warmups (`min(max(8, concurrency // 16), 32)`) to stabilize CUDA-graph captures before measurement.

* **Workload & Dataset Methodology**
  * **Uniform Concurrency & Prompts:** All evaluated models shared identical concurrency schedules and dynamic prompt sizing rules via `resolve_dynamic_num_prompts` (`max(concurrency * 2, base)`) to guarantee statistical stability without request starvation.
  * **Workload Restrictions (72B):** RAG and ultra-long context variants were completely excluded for the 72B model due to physical 32 GB VRAM saturation limits at 8K context.
  * **Timeout Safeguards:** Subprocess execution timeout extended to `timeout=7200` (2 hours) specifically to accommodate heavy single-concurrency coding workloads on the 72B model without premature termination (`rc=-1`).