# Steering Document: RLVF C++ → Rust Translation System

Design decisions, constraints, and rationale for navigating ambiguous choices during implementation.

---

## Architecture Decisions

### 1. Single-Turn vs Multi-Turn Episodes

**Decision: Start single-turn; add multi-turn only if compile rate plateaus below 40% after 200 steps.**

Single-turn (generate once, get reward) is far simpler to implement with GRPOTrainer. Multi-turn (generate → get compiler error → regenerate) requires custom environment logic and complicates credit assignment. The model's `<think>` block acts as a soft substitute for multi-turn — it can reason about potential errors before committing to code.

If compile rate doesn't reach 40% after 200 training steps, consider switching to a 2-turn setup: provide `rustc` error output as a second user message and allow one repair attempt.

### 2. Function Granularity

**Decision: Translate one function at a time. Never translate an entire file as a single episode.**

Function-level episodes give clean reward attribution, fit in context windows, and produce testable atomic units. File-level translation with 10+ functions is too long for completion and too coarse for reward signals.

When a function depends on another (e.g., `normalize` calls `norm`), include the already-translated Rust version of dependencies in the prompt context as read-only reference code — not something the model translates again.

### 3. Thinking Mode

**Decision: Always enable thinking mode (`enable_thinking=True`). Do not include `<think>` content in reward computation.**

Qwen3.5-2B benefits from chain-of-thought for code translation. The thinking block is free — it does not consume output token budget for reward purposes. Reward only the content after `</think>`.

If thinking blocks grow excessively long (> 1000 tokens for simple functions), add a soft length penalty on thinking tokens only.

### 4. Reward Weights

**Decision: 70% correctness / 20% latency / 10% memory. Do not tune these mid-training.**

These weights encode a strict priority: a fast-but-wrong translation is worse than a slow-but-correct one. The latency/memory rewards are only computed when all tests pass (see plan.md Phase 3.6), so they cannot override correctness.

Do not adjust weights based on early training behavior — premature tuning introduces noise. Re-evaluate only after a full training run.

### 5. No Reference Model (beta=0.0)

**Decision: Set `beta=0.0` (no KL divergence term). Do not load a reference model.**

The KL term was originally included in GRPO to prevent the policy from drifting too far from the base model. Recent work (DAPO, Open-Reasoner-Zero) shows it is unnecessary and wastes ~5GB VRAM. The clipping ratio `epsilon` already constrains the update magnitude.

If the model shows signs of reward hacking (generating trivially passing but non-functional code), add `beta=0.001` as a first corrective measure before investigating the reward function.

### 6. LoRA vs Full Fine-Tuning

**Decision: Use LoRA (r=16, alpha=16). Do not use QLoRA or full fine-tuning.**

Full fine-tuning Qwen3.5-2B requires ~24GB VRAM for the optimizer states alone. LoRA reduces this to ~14GB. QLoRA (4-bit + LoRA) is explicitly warned against for Qwen3.5 hybrid architecture due to the non-standard linear attention layers. Never use 4-bit loading.

### 7. C++ Library: Custom vs Off-the-Shelf

**Decision: Build a custom mini-library. Do not use a real-world library (Eigen, fmtlib, etc.) as training data.**

Real libraries contain: templates, preprocessor macros, platform guards, SIMD intrinsics, and complex memory patterns. These are beyond the model's reliable translation capability at 2B parameters and will produce noisy rewards. The custom library is designed for learnability, not production use.

The library should stay under 1500 LoC total to remain manageable. Each function must have at least 8 unit tests.

---

## Implementation Constraints

### Reward Function Constraints

- **Timeout**: Each reward evaluation must complete within 30 seconds. Set `subprocess` timeout=30 for compilation, timeout=60 for tests. If a timeout occurs, return -0.5 (same as compile failure).
- **Isolation**: Each evaluation runs in a fresh `tempfile.TemporaryDirectory()`. Never share state between reward evaluations — race conditions will silently corrupt rewards.
- **Idempotency**: The reward function must return the same value for the same input. Benchmark variance is the exception — see below.
- **Benchmark variance**: Only use latency reward when the variance across 5 runs is below 20%. Otherwise, return `None` from the latency reward function (TRL interprets `None` as "skip this reward component for this sample").

### Training Constraints

- **Never use `unsafe` in expected Rust output**: The library is designed to be translatable to safe Rust. If the reward function detects `unsafe` blocks, apply a penalty (subtract 0.2 from the final reward, floored at -0.5).
- **Max completion length 2048**: Rust translations of functions under 80 LoC should never exceed 2048 tokens. If a completion is truncated, `mask_truncated_completions=True` ensures it gets zero reward, not negative reward.
- **Checkpointing**: Save every 50 steps. Training Qwen3.5-2B can crash on long runs due to CUDA memory fragmentation. Resume from checkpoint — do not restart from scratch.

### Dataset Constraints

- **No data augmentation**: Do not paraphrase C++ source or vary prompts during GRPO training. The training signal comes from reward variance across generations, not prompt variance.
- **Fixed test cases**: Unit tests are frozen before training starts. Do not generate additional tests during training — this would change the reward landscape mid-training.
- **Dependency order**: Training examples within the same module must be presented in dependency order (leaf functions first). Shuffle only across modules, not within them.

---

## What to Do When Things Go Wrong

### Compile rate < 20% after 100 steps
The model is not learning to produce valid Rust syntax. Options in order:
1. Check prompt template — is the instruction clear? Does it specify the exact function signature?
2. Check tokenization — are code blocks being parsed correctly?
3. Add 5 SFT (supervised fine-tuning) warm-up steps on gold translations before GRPO starts
4. Reduce `max_completion_length` from 2048 to 1024 to reduce generation entropy

### Reward NaN
Caused by `std(rewards) = 0` in advantage normalization (all G generations got identical reward). Fix:
```python
# In custom advantage computation
std = torch.std(rewards) + 1e-4  # add epsilon before division
```
This is already handled by `scale_rewards="batch"` in most cases, but add the explicit epsilon as a safety net.

### Training loss spikes
Usually caused by a very high gradient norm on a single batch. Actions:
1. Add `max_grad_norm=1.0` to GRPOConfig
2. If spikes persist, reduce `epsilon_high` from 0.28 to 0.2 (symmetric clipping)
3. Reduce `learning_rate` from 1e-6 to 5e-7

### Model generates unsafe blocks despite penalty
The penalty (-0.2) is insufficient or the model has learned to game it. Actions:
1. Increase unsafe penalty to -0.5 (cap total reward at -0.5 for any unsafe usage)
2. Add `unsafe` to the negative instruction in the prompt: "You must not use `unsafe` blocks."
3. Filter unsafe completions from the advantage computation entirely (treat as compile failure)

### Rust translations are syntactically valid but semantically wrong
Unit tests are not catching bugs — test coverage is insufficient. Actions:
1. Add property-based tests using the `proptest` crate (generates random inputs)
2. Add differential testing: run C++ and Rust on identical inputs and compare outputs
3. Increase the number of test cases from 8 to 15+ per function

---

## Quality Gates (Before Declaring Success)

The system is working correctly when:

| Gate | Threshold |
|------|-----------|
| Compile rate on eval set | > 85% |
| Average test pass rate on eval set | > 0.75 |
| Functions achieving full test pass | > 60% |
| Functions with latency within 2x of C++ | > 80% of passing functions |
| Functions using `unsafe` | < 5% |
| Functions passing `cargo clippy` | > 70% |

These gates apply to the held-out eval set (20% of functions), evaluated after the full training run.

---

## Future Extensions (Out of Scope for v1)

- Multi-turn repair loop (RLEF-style)
- vLLM inference for faster generation (pending Qwen3.5 compatibility)
- Scaling to a real C++ library (would require function extraction tooling via clang LibTooling)
- Reward model distillation (training a critic to predict pass rates without execution)
- MCTS-guided generation for harder functions
