include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference
using Random

function run_shared_expert_experiment(; seed=42, n_train=500, n_val=100,
                                        epochs=30, lr=0.001, router_lr=0.0005,
                                        verbose=true)
    println("=" ^ 60)
    println("SHARED EXPERT EXPERIMENT")
    println("COMPRESS always active + router selects 1 from [RETRIEVE, REASON, PLAN]")
    println("=" ^ 60)

    println("\nGenerating datasets...")
    X_train, y_train, diff_train = generate_dataset(n_train; seed=seed)
    X_val, y_val, diff_val = generate_dataset(n_val; seed=seed + 100)

    model = create_model!(3; hidden_dim=16, memory_neurons=10, max_steps=5,
                          n_classes=6, seed=seed)

    println("Model created:")
    println("  Slices: $(length(model.slices)) (COMPRESS always active)")
    println("  Routed slices: $N_ROUTED_SLICES (RETRIEVE=1, REASON=2, PLAN=3)")
    println("  Slice output dim: $(model.slice_output_dim)")
    println("  Combined output dim: $(model.combined_output_dim)")
    println("  Output head: $(model.combined_output_dim) -> $(model.n_classes)")
    println("  Router slice head: $(size(model.router.slice_head_w))")
    println("  Max steps: $(model.max_steps)")

    train_loader = DataLoader(X_train, y_train, diff_train; batch_size=64)

     history = train_with_backprop!(model, train_loader, X_val, y_val, diff_val;
                                     epochs=epochs, lr=lr, router_lr=router_lr,
                                     print_every=5)

    metrics = evaluate(model, X_val, y_val, diff_val)
    println("\n" * "=" ^ 60)
    println("FINAL RESULTS")
    println("=" ^ 60)
    println("  Accuracy: $(round(metrics["accuracy"] * 100, digits=1))%")
    println("  Easy accuracy: $(round(metrics["easy_accuracy"] * 100, digits=1))%")
    println("  Medium accuracy: $(round(metrics["medium_accuracy"] * 100, digits=1))%")
    println("  Hard accuracy: $(round(metrics["hard_accuracy"] * 100, digits=1))%")
    println("  Avg steps (easy): $(round(metrics["easy_avg_steps"], digits=2))")
    println("  Avg steps (hard): $(round(metrics["hard_avg_steps"], digits=2))")
    println("  FLOPS saved: $(round(metrics["flops_saved"] * 100, digits=1))%")

    println("\n" * "=" ^ 60)
    println("ROUTING ANALYSIS")
    println("=" ^ 60)
    preds, traces = forward(model, X_val)
    slice_counts = zeros(Int, N_ROUTED_SLICES)
    phase_counts = zeros(Int, 4)
    for trace in traces
        for s in trace.slice_history
            slice_counts[s] += 1
        end
        for p in trace.phase_history
            phase_counts[Int(p)] += 1
        end
    end
    total_slice = sum(slice_counts)
    total_phase = sum(phase_counts)
    slice_names = ["RETRIEVE", "REASON", "PLAN"]
    for i in 1:N_ROUTED_SLICES
        pct = total_slice > 0 ? 100 * slice_counts[i] / total_slice : 0.0
        println("  $(slice_names[i]): $(round(pct, digits=1))% ($(slice_counts[i]) selections)")
    end
    phase_names = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]
    for i in 1:4
        pct = total_phase > 0 ? 100 * phase_counts[i] / total_phase : 0.0
        println("  Phase $(phase_names[i]): $(round(pct, digits=1))%")
    end

    return model, history, metrics
end

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(42)
    run_shared_expert_experiment(; epochs=30, verbose=true)
end
