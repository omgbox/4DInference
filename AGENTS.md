AGENTS — FourDInference (concise agent instructions)

High-signal facts an automated agent will likely miss.

Quick commands
- Instantiate deps (once): `julia --project=. -e "using Pkg; Pkg.instantiate()"`
- Run full test script (fast): `julia --project=. test/runtests.jl`
- Quick smoke example (short): `julia --project=. examples/quick_routing_check.jl`
- Long experiments are interactive/expensive: `julia --project=. examples/sequence_experiment.jl` (do not run on CI or without checking resources)

Entrypoints & layout
- Module entry: `src/FourDInference.jl` (examples/tests use `include(joinpath(@__DIR__, "..", "src", "FourDInference.jl")); using .FourDInference`).
- Library code lives under `src/*.jl` (data.jl, memory.jl, phase.jl, film.jl, slices.jl, router.jl, inference.jl, training.jl).
- Tests are a single script: `test/runtests.jl` — run it from the repository root with `--project=.`.

Environment & versions
- This project declares `julia = "1.10"` in Project.toml. Use `--project=.` for reproducible loading. If you change deps, run `Pkg.instantiate()`.

Behavioral gotchas (read before editing/running)
- Global state: examples and train scripts mutate globals (e.g. X_test_global, y_test_global, PhaseManager, Memory). Run examples/tests in a fresh Julia process to avoid cross-run contamination.
- Determinism: many tests/examples rely on explicit MersenneTwister seeds. Do not remove or change seeds when running tests.
- Heavy experiments: `examples/sequence_experiment.jl` and other training scripts run full experiments (hours on CPU). Use `examples/quick_routing_check.jl` or `examples/demo.jl` for quick verification.
- Module-loading pattern: code is not a registered package; keep the `include(...); using .FourDInference` pattern used in tests/examples. Relative paths matter (run from repo root).
- Do not add threaded JIT-heavy loops on Windows: on some Julia/LLVM combos (observed in workspace) adding `@threads` to inner numeric loops causes LLVM JIT crashes. Follow existing single-threaded inner loops.

CI notes
- CI should: `setup-julia` with 1.10, run `julia --project=. -e 'using Pkg; Pkg.instantiate()'`, then `julia --project=. test/runtests.jl`.
- Avoid running long experiments in CI. If you add CI that runs examples, limit them to short smoke checks only.

If something’s ambiguous
- Prefer the executable truth: code in `src/` and `test/runtests.jl` is the authoritative workflow. If docs conflict with code, follow code and update AGENTS.md.

Maintainer: omgbox
