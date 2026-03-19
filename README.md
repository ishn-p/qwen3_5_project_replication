# Qwen3.5-2B RLVF: C++ → Rust Translation

Training [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) to translate C++ library functions into idiomatic Rust using Reinforcement Learning from Verifiable Feedback (RLVF) with Group Relative Policy Optimization (GRPO).

---

## Overview

The model acts as an agent in a code translation environment. At each episode, it receives a C++ function and must produce a correct, performant Rust translation. Reward is computed automatically — no human labelers or reward models required.

```
Observation  →  C++ function source + context
Action       →  Rust translation (generated function-by-function)
Reward       →  Compilation success + unit test pass rate + latency + memory
Learning     →  GRPO (Group Relative Policy Optimization)
```

---

## Reward Structure

| Component | Weight | Signal |
|-----------|--------|--------|
| Unit test pass rate | 70% | `cargo nextest` JSON output |
| Latency vs C++ | 20% | Rust/C++ time ratio via Criterion |
| Memory vs C++ | 10% | Peak heap via `dhat` crate |
| Compile failure | — | −0.5 penalty (overrides above) |

Correctness dominates. Latency and memory rewards are only computed when all tests pass.

---

## Architecture

- **Model**: Qwen3.5-2B with LoRA (r=16, bf16, ~14GB VRAM during training)
- **Training**: TRL `GRPOTrainer` with `loss_type="dapo"`, `beta=0.0`, G=8 generations per prompt
- **Thinking mode**: Enabled — model reasons in `<think>` blocks before producing code
- **Environment**: Custom C++ mini-library (~25 functions) designed for learnability
- **Episode granularity**: One function per episode, translated in dependency order

---

## Project Structure

```
.
├── qwen3_5_rl.ipynb       # Main training notebook
├── .claude/
│   ├── plan.md            # Phased implementation plan
│   └── steering.md        # Design decisions and constraints
└── README.md
```

The full implementation plan is in [.claude/plan.md](.claude/plan.md). Design rationale and failure-mode playbook are in [.claude/steering.md](.claude/steering.md).

---

## Setup

```bash
# Python dependencies
pip install torch transformers trl peft datasets accelerate

# Rust toolchain
rustup install stable
cargo install cargo-nextest
```

Hardware: 1× A100 40GB minimum (or 2× 24GB with ZeRO-2).

> **Note**: Do not use 4-bit quantization (QLoRA) with Qwen3.5 — the hybrid linear attention architecture has known incompatibilities.

---

## References

- [DeepSeek-R1](https://arxiv.org/pdf/2501.12948) — GRPO at scale
- [DAPO](https://arxiv.org/pdf/2503.14476) — Loss type used here
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Syzygy](https://arxiv.org/abs/2412.14234) — Function-level C→Rust translation
- [CodeRL](https://arxiv.org/abs/2207.01780) — RL for code generation
