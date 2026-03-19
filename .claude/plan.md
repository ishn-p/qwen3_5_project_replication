# RLVF System: Qwen3.5-2B C++ → Rust Translation
## Implementation Plan

---

## Project Overview

Train Qwen3.5-2B using GRPO (Group Relative Policy Optimization) to translate C++ library functions into idiomatic Rust. The model receives a C++ function as observation and generates a Rust translation as action. Reward is computed from automated evaluation: compilation, unit tests, latency, and memory.

---

## Phase 0: Environment Setup

### 0.1 Dependencies
```
torch >= 2.3.0 (CUDA 12.1+)
transformers >= 4.51.0
trl >= 0.17.0
peft >= 0.15.0
unsloth (optional but recommended for Qwen3.5)
datasets
accelerate
```

Rust toolchain:
```bash
rustup install stable
cargo install cargo-nextest
cargo install cargo-criterion  # optional latency benchmarks
```

### 0.2 Hardware Requirements
- Minimum: 1x A100 40GB (or 2x 24GB with ZeRO-2)
- Recommended: 1x A100 80GB
- With LoRA r=16, bf16, G=8: ~14GB peak VRAM during training

---

## Phase 1: C++ Training Library Construction

### 1.1 Design the `cpp_training_lib`

Create a purpose-built C++ library (~25 functions across 4 modules). Each function must be:
- Self-contained (no external deps beyond `<cmath>`, `<vector>`, `<string>`)
- Under 80 LoC
- Testable with numeric/string I/O
- Translatable to safe Rust (no raw pointer arithmetic, no manual memory management)

**Directory structure:**
```
cpp_training_lib/
├── CMakeLists.txt
├── include/
│   ├── math_ops.h
│   ├── container_ops.h
│   ├── string_ops.h
│   └── algorithm_ops.h
├── src/
│   ├── math_ops.cpp       # dot, cross, norm, normalize, lerp, clamp
│   ├── container_ops.cpp  # ring_buffer (push/pop/peek), sorted insert, merge
│   ├── string_ops.cpp     # split, join, trim, pad, count_substr, reverse_words
│   └── algorithm_ops.cpp  # binary_search variants, quicksort, run_length_encode
└── tests/
    ├── test_math.cpp
    ├── test_container.cpp
    ├── test_string.cpp
    └── test_algorithm.cpp
```

### 1.2 Function Catalog (25 functions)

| Module | Functions | Rust Complexity |
|--------|-----------|-----------------|
| math_ops | dot_product, cross_product, vec_norm, normalize, lerp, clamp, sigmoid | Low |
| container_ops | ring_buffer::push/pop/peek/is_full, sorted_insert, merge_sorted | Medium |
| string_ops | split, join, trim, pad_left/right, count_substr, reverse_words | Low |
| algorithm_ops | binary_search, lower_bound, upper_bound, quicksort, run_length_encode | Medium |

### 1.3 Test Harness (per function, 8 tests minimum)
- 3 typical cases
- 2 edge cases (empty input, single element, boundary values)
- 2 stress cases (large inputs for timing)
- 1 negative case (invalid input handling)

Auto-generate Rust `#[test]` equivalents by translating test inputs/outputs directly (simpler than translating the library itself).

---

## Phase 2: Dataset Construction

### 2.1 Dataset Schema
```python
{
    "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(**fn_data)}],
    "cpp_source": str,          # full C++ function body
    "function_name": str,       # e.g. "dot_product"
    "module": str,              # e.g. "math_ops"
    "test_cases": list[dict],   # [{inputs: ..., expected: ...}, ...]
    "rust_test_src": str,       # pre-written Rust test code
    "cpp_signature": str,       # C++ function signature
    "rust_signature": str,      # expected Rust signature (for validation)
    "difficulty": int,          # 1=easy, 2=medium, 3=hard (for curriculum)
}
```

### 2.2 Prompt Template
```
You are translating a C++ function to idiomatic Rust.

## C++ Source
```cpp
{cpp_source}
```

## Requirements
- Translate to safe Rust (no `unsafe` blocks unless absolutely necessary)
- Use idiomatic Rust: iterators, pattern matching, `Result`/`Option` where appropriate
- The function signature should be: `{rust_signature}`
- Do not include `use` statements or module declarations — only the function body

## Output
Provide your translation inside a ```rust code block.
```

### 2.3 Curriculum Ordering
Sort training examples by `difficulty` (1→3). Within each difficulty level, order by dependency (leaf functions first in the call graph). This implements a simple curriculum that avoids frustrating early episodes.

---

## Phase 3: Reward Functions

### 3.1 Compilation Reward (`reward_compile`)
```
compile_success  → 0.0  (neutral, not rewarded for just compiling)
compile_failure  → -0.5 (penalty, but not -1 to preserve gradient signal)
```

### 3.2 Test Pass Reward (`reward_tests`)
```
pass_rate = passed_tests / total_tests   ∈ [0.0, 1.0]
reward = pass_rate
```
Implementation: parse `cargo nextest run --message-format libtest-json-plus` output.

### 3.3 Latency Reward (`reward_latency`)
```
ratio = rust_time_ns / cpp_time_ns
reward = clamp(1.0 - (ratio - 0.5), 0.0, 1.0)
  → ratio=0.5 (Rust 2x faster) → reward=1.0
  → ratio=1.0 (parity)         → reward=0.5
  → ratio=2.0 (Rust 2x slower) → reward=0.0
```
Only compute if tests pass (skip overhead when code is incorrect).

### 3.4 Memory Reward (`reward_memory`)
```
ratio = rust_peak_bytes / cpp_peak_bytes
reward = clamp(2.0 - ratio, 0.0, 1.0)
```
Use `dhat` crate for cross-platform heap tracking.

### 3.5 Composite Reward
```
R = 0.7 * reward_tests + 0.2 * reward_latency + 0.1 * reward_memory
  + penalty_compile (applied before component rewards if compile fails)
```

Weight rationale: correctness dominates; performance is secondary; never reward fast-but-wrong code.

### 3.6 Reward Execution Pipeline
```python
async def evaluate_completion(rust_code: str, fn_data: dict) -> float:
    with TempCargoProject(rust_code, fn_data["rust_test_src"]) as proj:
        compile_ok = await build(proj)
        if not compile_ok:
            return -0.5

        test_result = await run_tests(proj)
        pass_rate = test_result["pass_rate"]

        if pass_rate < 1.0:
            return pass_rate * 0.7  # partial credit, skip perf

        latency = await run_benchmark(proj, fn_data)
        memory = await run_memory_profile(proj, fn_data)

        return 0.7 + 0.2 * latency_reward(latency) + 0.1 * memory_reward(memory)
```

---

## Phase 4: GRPO Training Setup

### 4.1 Model Loading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # if available
)
```

Do NOT use 4-bit quantization (QLoRA) — Qwen3.5 hybrid architecture has known issues.

### 4.2 LoRA Configuration
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,        # alpha = r (not 2r) for Qwen3.5
    lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
```

### 4.3 GRPOConfig
```python
from trl import GRPOConfig

config = GRPOConfig(
    output_dir="./checkpoints/qwen35-cpp2rust",
    learning_rate=1e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    bf16=True,
    gradient_checkpointing=True,

    # GRPO
    num_generations=8,
    max_completion_length=2048,
    temperature=0.8,
    top_p=0.95,
    top_k=20,

    # Loss
    loss_type="dapo",
    scale_rewards="batch",
    beta=0.0,
    epsilon=0.2,
    epsilon_high=0.28,

    # Thinking
    chat_template_kwargs={"enable_thinking": True},

    # Misc
    mask_truncated_completions=True,
    log_completions=True,
    remove_unused_columns=False,
    logging_steps=10,
    save_steps=100,
)
```

### 4.4 Reward Functions Registration
Register three separate reward functions with `reward_weights=[0.7, 0.2, 0.1]` in the trainer. This lets TRL log each reward component separately for debugging.

---

## Phase 5: Evaluation & Monitoring

### 5.1 Training Metrics to Watch
| Metric | Target | Problem if... |
|--------|--------|---------------|
| `reward/mean` | Trending up | Flat after 200 steps → check reward function |
| `frac_reward_zero_std` | < 0.3 | High → all gens identical → increase G or temperature |
| `completions/clipped_ratio` | < 0.1 | High → max_completion_length too short |
| `entropy` | Slow decrease | Crash → reduce epsilon_high or lr |
| `compile_rate` | > 0.3 early, 0.8+ late | Low → prompt engineering needed |

### 5.2 Held-Out Evaluation Set
Reserve 20% of functions as eval set. Run full evaluation pipeline (compile + test + benchmark) every 50 training steps on the eval set. Report:
- Compilation success rate
- Average test pass rate
- % of functions achieving parity or better latency vs C++

### 5.3 Qualitative Review
Every 100 steps, log 5 random completions with their rewards. Look for:
- Unsafe block usage (should trend to 0)
- Idiomatic Rust patterns (iterators vs manual loops)
- Thinking quality in `<think>` blocks

---

## Phase 6: Notebook Structure (`qwen3_5_rl.ipynb`)

The notebook is organized into self-contained cells:

```
Cell 1:  Setup & Imports
Cell 2:  C++ Library Loading + Dependency Graph
Cell 3:  Dataset Construction
Cell 4:  Reward Function Implementations (compile, test, latency, memory)
Cell 5:  Reward Unit Tests (verify reward functions work correctly)
Cell 6:  Model + LoRA Setup
Cell 7:  GRPOConfig + Trainer Construction
Cell 8:  Training Loop
Cell 9:  Evaluation & Metrics Dashboard
Cell 10: Inference Demo (translate a new function interactively)
```

---

## Implementation Order

1. **Week 1**: Build `cpp_training_lib`, write C++ tests, verify tests pass in C++
2. **Week 1**: Implement and unit-test reward functions (mock completions)
3. **Week 2**: Construct dataset, verify prompt/tokenization pipeline
4. **Week 2**: Set up GRPO training loop, run 50-step smoke test
5. **Week 3**: Full training run (3 epochs), monitor metrics
6. **Week 3**: Evaluation, qualitative review, hyperparameter tuning
7. **Week 4**: Polish notebook, document results

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Sparse rewards early (all compile failures) | Curriculum ordering by difficulty; warm start with SFT on 5 examples |
| Reward NaN from zero std | `std = max(std(r), 1e-4)` in advantage computation |
| Context overflow (long C++ functions) | Cap functions at 200 LoC; trim context to 2048 tokens |
| Qwen3.5 vLLM compatibility | Use HF generate, not vLLM, until confirmed compatible |
| Benchmark noise in latency reward | Run 5 benchmark iterations; use median; skip latency reward if variance > 20% |
| Unsafe Rust reward hacking | Add unsafe block penalty to reward function |
