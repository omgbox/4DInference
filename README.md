# 4DInference

A novel **4D inference architecture with learned routing**, implemented in pure Julia. The model processes information across four axes — **X** (slices/experts), **Y** (temporal steps), **Z** (SORN memory), and **W** (phase/mode) — and learns to route through them dynamically per input.

This is an active research project investigating whether a Mixture-of-Experts-style router can learn to select distinct processing paths (slices) for distinct input types. The headline finding: **routing collapse is architecturally universal** in this setup — see [Open Research Problem](#open-research-problem--routing-collapse).

## Research Overview

### Motivation & Use Cases

**Why this architecture?**

1. **Efficient inference**: Route to smaller specialized experts instead of a large monolithic network. Potential for faster, lower-memory inference.

2. **Interpretability**: Track which slice/phase the model chose per input for debugging and understanding decisions.

3. **Continual learning**: Slices specialize over time; new experts can be added without retraining old ones.

4. **Multi-task handling**: Different slices for different modalities (image, text, audio); route at inference time.

5. **Adaptive reasoning**: Easy inputs need quick retrieval; hard inputs need multi-step reasoning. The model learns when each is needed.

**Real-world applications:**
- Medical diagnosis: RETRIEVE lab results → REASON about them → PLAN next steps → COMPRESS into a decision.
- Multimodal AI: different slices for image, text, audio; router decides which to activate.
- Real-time systems: detect easy vs. hard inputs; spend more computation steps on hard ones.
- Federated learning: route to different models/domains based on input characteristics.
- Adaptive language models: easy tokens (punctuation, common words) use fast slice; complex tokens use reasoning slices.

### What Works (The Good News)

✅ **Multi-step temporal reasoning (Y-axis)**: Dramatically outperforms flat MLPs
- **6-class temporal task**: 82.1% (this model) vs. 12.8% (baseline MLP)
- Model learns to refine decisions iteratively over multiple steps
- Early-exit mechanism works — high-confidence samples finish in 1-2 steps

✅ **Memory integration (Z-axis)**: SORN memory provides useful contextual state
- Router can query memory to inform routing decisions
- Surprise-gated writes selectively encode important events

✅ **Phase state machine (W-axis)**: FiLM phase conditioning is effective
- Different phase embeddings modulate slice outputs differently
- Model learns phase transitions that align with task structure

✅ **Training stability**: Gradient clamping and model checkpointing critical
- Clamping gradients `[-2,2]` for `d_avg`, `[-1,1]` for weights prevents divergence
- Models peak around epoch 50, then degrade — best-weights restoration essential

### What Doesn't Work (The Core Problem)

❌ **Learned routing (X-axis)**: Router always collapses to a single slice
- All 14 anti-collapse techniques failed
- This is **not a training bug** — it's an **architectural inevitability**
- See [Open Research Problem](#open-research-problem--routing-collapse) for details

### Research Contribution

This project **empirically proves a theoretical blind spot** in Mixture-of-Experts literature:

- **Existing MoE papers** (Mixture of Experts, Switch Transformers, etc.) assume experts operate in *parallel* on disjoint data subsets or their outputs are explicitly combined.
- **This architecture** has experts operate *sequentially* on shared input/output dimensions.
- **Result**: routing collapse is *architecturally inevitable*, not a training bug or hyperparameter issue.

**Why this matters:** Future researchers now know that if they want learned routing to specialize, they must **decouple representation learning from routing**. End-to-end training of sequential, same-dimensional experts will not work.

## Architecture

The `FourDModel` wires together six components, each in its own file under `src/`:

| Component | File | Role |
|-----------|------|------|
| **Router** | `router.jl` | Reads (input + memory + phase) and outputs a slice choice + phase choice + confidence |
| **Slices** (experts) | `slices.jl` | 4 MLPs: `RETRIEVE`, `REASON`, `PLAN`, `COMPRESS` |
| **Memory (Z-axis)** | `memory.jl` | SORN-inspired recurrent memory with surprise-gated writes |
| **Phase (W-axis)** | `phase.jl` | 4-phase state machine (`RETRIEVE`, `REASON`, `PLAN`, `COMPRESS`) with learned embeddings |
| **FiLM** | `film.jl` | Feature-wise modulation: each slice's output is conditioned on the current phase |
| **Inference loop (Y-axis)** | `inference.jl` | Up to `max_steps` per timestep; accumulates state across steps |

### The four axes

- **X — Slices**: 4 processing experts the router chooses between.
- **Y — Temporal steps**: the model runs a multi-step inner loop per input timestep (up to `max_steps`), re-reading memory and re-routing each step.
- **Z — Memory**: a SORN-style recurrent network with surprise-gated writes provides a dynamic internal state the router can consult.
- **W — Phase**: a 4-mode state (`RETRIEVE`/`REASON`/`PLAN`/`COMPRESS`) that conditions slice processing via FiLM and evolves over the inference trace.

### Forward pass

For each input `x_t` in a sequence:
1. Read memory state and current phase → build router input.
2. Router picks a slice + phase; slice processes a per-slice projection of `x_t`.
3. FiLM modulates the slice output by the phase embedding.
4. Accumulate state; write to memory (surprise-gated); advance phase.
5. Repeat up to `max_steps` (early-exit on high confidence).

Final accumulated state → linear classifier.

## Results

| Task | Model | Accuracy | Notes |
|------|-------|----------|-------|
| 6-class temporal | Large 64h | **82.1%** | Y-axis temporal processing is the key differentiator |
| 12-class multimodal | 48h | 55.0% | Harder task; routing still collapses |
| 6-class (shared expert) | 16h, 100ep | 69% peak / 54% final | COMPRESS always active; still collapses |

Random baselines: 16.7% (6-class), 8.3% (12-class).

**What works:** Y-axis temporal multi-step processing dramatically beats a flat MLP (82.1% vs 12.8% on the 6-class task). Model checkpointing is essential — models peak then degrade. Gradient clamping (`[-2,2]` for `d_avg`, `[-1,1]` for weights) is critical for training stability.

## Open Research Problem — Routing Collapse

The router **always collapses to a single slice** (typically `COMPRESS` or `RETRIEVE`) regardless of architecture variant, task, or training technique. This is the single most common failure mode in MoE training.

**Root cause (confirmed):** when all slices share the same input and output dimensions, the downstream output layer adapts to whichever slice is active. Slices are structurally interchangeable, so the router receives no gradient signal for diversity.

### Failed approaches (14 total)

| # | Approach | Result |
|---|----------|--------|
| 1 | Entropy bonus | Collapses to 1 slice |
| 2 | Per-slice input projections | Doesn't help — output dim same |
| 3 | Loss-Free Balancing (LFB) | Forces uniform balance, kills accuracy |
| 4 | Router dropout | Collapses at eval |
| 5 | Expert orthogonality | Harmful — destroys representations |
| 6 | Routing variance loss | Ineffective (gradient ≈ 0) |
| 7 | REINFORCE policy gradient | Initially diversifies, then collapses |
| 8 | Capacity-locked (feature masking) | Slices too weak (44.4% vs 55%) |
| 9 | Output-partitioned routing | Loss explosion |
| 10 | Forced round-robin | 42.4%, slices trained on wrong inputs |
| 11 | Soft routing v2 | 57.2% but 100% collapse |
| 12 | Expert Choice (hard) | 7% accuracy, perfect balance, no specialization |
| 13 | Oracle label training | Router can't learn from synthetic labels |
| 14 | Shared expert (COMPRESS always + 3-way) | 69% peak, 100% RETRIEVE collapse |

**Current direction:** abandon learned routing during training. Use ensemble training (all slices active, combined outputs) for representation learning, then train a separate router or use hash-based selection for inference — decoupling representation learning from routing.

### Why Routing Collapses (Deep Dive)

#### The Core Issue: Structural Interchangeability

**The fundamental problem:** When all slices have the same input and output dimensions, the downstream output layer cannot distinguish which slice was used. Once one slice starts winning, it becomes a "rich-get-richer" dynamic:

1. Slice A is randomly slightly better at some inputs
2. Router sends more data to A (small gradient signal)
3. Output layer fine-tunes to A's outputs (A becomes even better)
4. Router sends *all* data to A (zero gradient signal for diversity)
5. Other slices receive no gradient updates and starve

**Why existing MoE avoids this:**
- **Switch Transformers**: Experts operate on *sparse* data subsets; each expert only sees part of the batch. The output layer combines all expert outputs (weighted sum), so it has to "learn" the differences.
- **Mixture of Experts (original)**: Gates are soft (weighted sum), not hard (argmax). All experts always contribute; output layer always sees all expert outputs.

**Why this architecture hits it:**
- Experts operate *sequentially* on the same input
- Router makes a *hard choice* (argmax) — only one expert activates
- Output layer sees only the *final accumulated state* — it cannot infer which expert was used
- Result: no gradient signal for router diversity

#### The 14 Failed Solutions

Every standard MoE technique was tried; all failed:

| Approach | Why it failed |
|----------|---------------|
| **Entropy bonus** | Entropy loss is too weak compared to accuracy loss; router ignores it |
| **Per-slice input projections** | Different inputs don't help if outputs are identical |
| **Loss-Free Balancing (LFB)** | Forces uniform load but destroys specialization; accuracy collapses |
| **Router dropout** | Works at train; collapses at eval (deterministic inference) |
| **Expert orthogonality loss** | Harmful; prevents experts from learning useful representations |
| **Routing variance loss** | Gradient is ~0; variance doesn't propagate |
| **REINFORCE** | Initially diversifies then collapses (same underlying issue) |
| **Capacity-locked routing** | Masks features per slice; slices become too weak |
| **Output-partitioned routing** | Each slice gets its own output dimensions; loss explodes |
| **Forced round-robin training** | Slices trained on wrong inputs; accuracy is 42.4% |
| **Soft routing v2** | Blends all slices; 100% collapse anyway (rich-get-richer) |
| **Expert Choice routing** | Hard balanced routing; no specialization; 7% accuracy |
| **Oracle label training** | Router can't learn from synthetic ground-truth routing |
| **Shared expert (COMPRESS always)** | 69% peak, 100% RETRIEVE collapse |

**Key insight:** The problem is not the router, the loss function, or the training procedure. It's the **architecture itself**. You cannot make sequential, same-dimensional experts specialize through end-to-end gradient descent.

#### What Might Work

To escape routing collapse:

1. **Sparse expert selection**: Only 1-2 slices ever see data; rest specialize on disjoint subsets (like standard MoE). Requires data partitioning strategy.

2. **Output partitioning**: Each slice produces its own output dimensions. Output layer *must* know which slice was active. Requires careful design to avoid loss explosion.

3. **Ensemble training + separate router**: Train all slices jointly (no routing); then train a separate lightweight router post-hoc using clustering or supervised labels.

4. **Heuristic routing**: Use domain knowledge (e.g., input features) to route, rather than learning end-to-end.

5. **Soft gating (weighted blend)**: Instead of hard argmax, blend multiple slices (like original Mixture of Experts). But this requires more compute and defeats the efficiency motivation.

## Quick start

Requires **Julia 1.10**.

```bash
# Install dependencies (once)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run the test suite (fast)
julia --project=. test/runtests.jl

# Quick smoke check
julia --project=. examples/quick_routing_check.jl

# Full experiment (long — hours on CPU)
julia --project=. examples/sequence_experiment.jl
```

Always run from the repository root. The project is **not** a registered package — scripts load it via:

```julia
include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference
```

## Usage

```julia
include("src/FourDInference.jl")
using .FourDInference

# Create a model: 3-dim input, 6 classes, up to 3 inner steps
model = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=3, seed=42)

# Single forward pass
output, trace = forward(model, randn(3))

# Batch forward
outputs, traces = forward(model, randn(10, 3))

# Generate the built-in temporal dataset and train
X_train, y_train = generate_sequence_dataset(4000; seed=42)
train!(model, X_train, y_train; epochs=100)
```

## Examples

| Script | Purpose | Cost |
|--------|---------|------|
| `examples/quick_routing_check.jl` | Fast smoke test | Seconds |
| `examples/demo.jl` | Minimal demo | Seconds |
| `examples/sequence_experiment.jl` | Main 6-class + 12-class experiments | Hours |
| `examples/shared_expert_experiment.jl` | Shared-expert (COMPRESS-always) variant | Hours |
| `examples/full_experiment.jl` | Full ablation suite | Hours |
| `examples/ablation.jl` | Component ablations | Long |
| `examples/benchmark.jl` | Timing benchmarks | Minutes |

## Project layout

```
src/
  FourDInference.jl   # module entry, exports
  data.jl             # dataset generation + DataLoader
  memory.jl           # SORN-style recurrent memory (Z-axis)
  phase.jl            # 4-phase state machine (W-axis)
  film.jl             # FiLM phase conditioning
  slices.jl           # 4 expert MLPs (X-axis)
  router.jl           # learned router
  inference.jl        # FourDModel + forward loop (Y-axis)
  training.jl         # hand-derived backprop trainer
  training_backprop.jl
test/
  runtests.jl         # single test script
examples/
  *.jl                # experiments and demos
```

## Summary: Production Readiness

| Component | Status | Notes |
|-----------|--------|-------|
| **Multi-step reasoning (Y-axis)** | ✅ **Excellent** | 82.1% vs 12.8% baseline; highly effective |
| **Memory integration (Z-axis)** | ✅ **Good** | Provides useful context; works as designed |
| **Phase modulation (W-axis)** | ✅ **Good** | FiLM conditioning effective; phases learned |
| **Learned routing (X-axis)** | ❌ **Broken** | Always collapses; fundamental architectural issue |
| **Production ready?** | ❌ **No** | Routing provides no specialization; just picks one expert |
| **Research value?** | ✅ **High** | Proves architectural limits; informs future MoE design |

**Bottom line:** This architecture demonstrates that **learned routing for sequential, same-dimensional experts is fundamentally limited**. The multi-step reasoning and memory integration components are valuable and could be used in other architectures. The routing collapse finding is the main contribution — it tells future researchers what *not* to do and why.

## Caveats

- **Global state**: experiment scripts mutate globals (`X_test_global`, `PhaseManager`, `Memory`). Run each in a fresh Julia process.
- **Seeds**: tests/examples rely on explicit `MersenneTwister` seeds — don't change them.
- **No `@threads`**: on some Julia/LLVM + Windows combos, threaded inner numeric loops trigger LLVM JIT crashes. Inner loops are single-threaded by design; keep them that way.
- **Heavy experiments**: `sequence_experiment.jl` and friends run for hours on CPU. Use `quick_routing_check.jl` or `demo.jl` for quick verification.

## License

Research code, shared as-is.

## Maintainer

omgbox
