using Random
using Statistics
using Dates

include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference

function print_header(title::String)
    width = 70
    println("\n" * "=" ^ width)
    println(" " ^ max(0, div(width - length(title), 2)) * title)
    println("=" ^ width)
end

function count_parameters(model::FourDModel)
    total = 0
    total += length(model.input_proj_w) + length(model.input_proj_b)
    total += length(model.output_w) + length(model.output_b)
    for slice in model.slices
        for W in slice.weights
            total += length(W)
        end
        for b in slice.biases
            total += length(b)
        end
    end
    for film in model.film_layers
        total += length(film.gamma_w) + length(film.gamma_b)
        total += length(film.beta_w) + length(film.beta_b)
    end
    total += length(model.router.W1) + length(model.router.b1)
    total += length(model.router.W2) + length(model.router.b2)
    total += length(model.router.slice_head_w) + length(model.router.slice_head_b)
    total += length(model.router.phase_head_w) + length(model.router.phase_head_b)
    total += length(model.router.confidence_head_w) + 1
    return total
end

function main()
    print_header("FULL EXPERIMENT — 4D vs MLP WITH ALL IMPROVEMENTS")
    println("  Started: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")

    println("""
    Improvements being tested:
    1. 10k+ training samples
    2. Curriculum training (easy → medium → hard → mixed)
    3. Stronger entropy regularization (0.15 → 0.45)
    4. Sequence data support
    """)

    N_TRAIN = 5000
    N_VAL = 1000
    N_TEST = 2000
    EPOCHS = 40

    println("Dataset sizes: train=$N_TRAIN, val=$N_VAL, test=$N_TEST")
    println("Epochs per phase: $EPOCHS")
    println()

    results = []

    # ── Experiment 1: 4D with curriculum training ──
    print_header("EXPERIMENT 1: 4D + Curriculum Training")
    model_4d = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=3, seed=42)
    params_4d = count_parameters(model_4d)
    println("  Parameters: $params_4d")

    X_train, y_train, diff_train = generate_dataset(N_TRAIN; seed=42)
    X_val, y_val, diff_val = generate_dataset(N_VAL; seed=123)
    X_test, y_test, diff_test = generate_dataset(N_TEST; seed=456)

    baseline_4d = evaluate(model_4d, X_test, y_test, diff_test)
    println("  Baseline accuracy: $(round(baseline_4d["accuracy"], digits=3))")

    train_loader = DataLoader(X_train, y_train, diff_train; batch_size=64)
    h_4d = train_with_backprop!(model_4d, train_loader, X_val, y_val, diff_val;
                                epochs=EPOCHS * 3, lr=0.001, router_lr=0.0005,
                                entropy_coeff=0.15, print_every=EPOCHS)

    final_4d = evaluate(model_4d, X_test, y_test, diff_test)
    push!(results, ("4D + Backprop", params_4d, baseline_4d["accuracy"], final_4d, h_4d))

    # ── Experiment 2: 4D with curriculum training ──
    print_header("EXPERIMENT 2: 4D + Full Curriculum")
    model_curr = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=3, seed=42)
    params_curr = count_parameters(model_curr)
    println("  Parameters: $params_curr")

    h_curr = train_curriculum!(model_curr, N_TRAIN ÷ 3;
                               epochs_per_phase=EPOCHS, lr=0.001, router_lr=0.0005,
                               entropy_coeff=0.15, seed=42)

    final_curr = evaluate(model_curr, X_test, y_test, diff_test)
    push!(results, ("4D + Curriculum", params_curr, 0.155, final_curr, h_curr))

    # ── Experiment 3: MLP baseline ──
    print_header("EXPERIMENT 3: MLP Baseline (single slice)")
    model_mlp = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=1, seed=42)
    params_mlp = count_parameters(model_mlp)
    println("  Parameters: $params_mlp")

    baseline_mlp = evaluate(model_mlp, X_test, y_test, diff_test)
    println("  Baseline accuracy: $(round(baseline_mlp["accuracy"], digits=3))")

    train_loader_mlp = DataLoader(X_train, y_train, diff_train; batch_size=64)
    h_mlp = train_with_backprop!(model_mlp, train_loader_mlp, X_val, y_val, diff_val;
                                 epochs=EPOCHS * 3, lr=0.001, router_lr=0.0005,
                                 entropy_coeff=0.15, print_every=EPOCHS)

    final_mlp = evaluate(model_mlp, X_test, y_test, diff_test)
    push!(results, ("MLP Baseline", params_mlp, baseline_mlp["accuracy"], final_mlp, h_mlp))

    # ── Experiment 4: 4D with high entropy ──
    print_header("EXPERIMENT 4: 4D + High Entropy (prevents collapse)")
    model_ent = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=3, seed=42)
    params_ent = count_parameters(model_ent)
    println("  Parameters: $params_ent")

    train_loader_ent = DataLoader(X_train, y_train, diff_train; batch_size=64)
    h_ent = train_with_backprop!(model_ent, train_loader_ent, X_val, y_val, diff_val;
                                 epochs=EPOCHS * 3, lr=0.001, router_lr=0.001,
                                 entropy_coeff=0.4, print_every=EPOCHS)

    final_ent = evaluate(model_ent, X_test, y_test, diff_test)
    push!(results, ("4D + High Entropy", params_ent, 0.155, final_ent, h_ent))

    # ═══════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════
    print_header("FINAL RESULTS")

    println("\n  Model                 │ Params │ Base  │ Final │ Easy  │ Hard  │ Steps │ FLOPs")
    println("  " * "─" ^ 85)

    for (name, params, base, final, hist) in results
        name_p = rpad(name, 22)
        p = rpad(string(params), 6)
        b = round(base, digits=3)
        f = round(final["accuracy"], digits=3)
        e = round(final["easy_accuracy"], digits=3)
        h = round(final["hard_accuracy"], digits=3)
        s = round(final["easy_avg_steps"], digits=1)
        fl = round(final["flops_saved"], digits=3)
        improvement = f - b
        marker = improvement > 0.3 ? " ★" : ""
        println("  $name_p │ $p │ $b  │  $f  │ $e  │ $h  │  $s   │  $fl$marker")
    end

    println("  " * "─" ^ 85)
    println("  ★ = >30% improvement")

    # Winner analysis
    print_header("WINNER ANALYSIS")

    best_idx = findmax(r -> r[4]["accuracy"], results)[2]
    best_name = results[best_idx][1]
    best_acc = results[best_idx][4]["accuracy"]
    best_flops = results[best_idx][4]["flops_saved"]

    mlp_idx = findfirst(r -> occursin("MLP", r[1]), results)
    mlp_acc = results[mlp_idx][4]["accuracy"]

    fourd_idx = findfirst(r -> occursin("Curriculum", r[1]), results)
    fourd_acc = results[fourd_idx][4]["accuracy"]

    println("""
    Best overall: $best_name ($(round(best_acc*100, digits=1))% accuracy)

    4D vs MLP comparison:
      MLP accuracy: $(round(mlp_acc*100, digits=1))%
      4D (curriculum) accuracy: $(round(fourd_acc*100, digits=1))%
      Difference: $(round((fourd_acc - mlp_acc)*100, digits=1))%

    If 4D > MLP: The 4D architecture wins with proper training!
    If 4D < MLP: Need more data/epochs or architecture changes.
    """)

    # Routing analysis for best 4D model
    print_header("ROUTING ANALYSIS — CURRICULUM MODEL")

    slice_counts = zeros(4)
    phase_counts = zeros(4)
    step_counts = Float64[]
    easy_steps = Float64[]
    hard_steps = Float64[]

    for i in 1:min(200, size(X_test, 1))
        trace = forward(model_curr, @view X_test[i, :])
        for s in trace.slice_history
            slice_counts[s] += 1
        end
        for p in trace.phase_history
            phase_counts[Int(p)] += 1
        end
        push!(step_counts, trace.total_steps)
        if diff_test[i] == 1
            push!(easy_steps, trace.total_steps)
        elseif diff_test[i] == 3
            push!(hard_steps, trace.total_steps)
        end
    end

    slice_labels = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]
    phase_labels = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]

    println("\n  Slice Usage (should be diverse):")
    for (count, label) in zip(slice_counts, slice_labels)
        pct = count / sum(slice_counts) * 100
        bar = "█" * ("░" ^ round(Int, pct / 3))
        println("    $(rpad(label, 10)) $(round(pct, digits=1))% $bar")
    end

    println("\n  Phase Usage (should be diverse):")
    for (count, label) in zip(phase_counts, phase_labels)
        pct = count / sum(phase_counts) * 100
        bar = "█" * ("░" ^ round(Int, pct / 3))
        println("    $(rpad(label, 10)) $(round(pct, digits=1))% $bar")
    end

    slice_entropy = 0.0
    for c in slice_counts
        if c > 0
            p = c / sum(slice_counts)
            slice_entropy -= p * log(p)
        end
    end
    max_entropy = log(4)
    norm_entropy = slice_entropy / max_entropy

    println("\n  Routing diversity: $(round(norm_entropy * 100, digits=1))% (100% = uniform)")
    println("  Easy samples: $(length(easy_steps)) | mean steps: $(round(mean(easy_steps), digits=2))")
    println("  Hard samples: $(length(hard_steps)) | mean steps: $(round(mean(hard_steps), digits=2))")
    println("  Adaptive? $(length(easy_steps) > 0 && abs(mean(easy_steps) - mean(hard_steps)) > 0.5 ? "YES" : "no")")

    println("\n  Steps per sample: $(round(mean(step_counts), digits=2)) mean, " *
            "$(round(minimum(step_counts), digits=0)) min, $(round(maximum(step_counts), digits=0)) max")

    # Convergence analysis
    print_header("CONVERGENCE ANALYSIS")

    for (name, _, _, _, hist) in results
        if length(hist["loss"]) > 0
            losses = hist["loss"]
            println("\n  $(name):")
            println("    Start: $(round(losses[1], digits=4))")
            if length(losses) >= 10
                println("    Mid:   $(round(losses[end÷2], digits=4))")
            end
            println("    End:   $(round(losses[end], digits=4))")
            println("    Δ:     $(round(losses[1] - losses[end], digits=4))")
        end
    end

    println("\n" * "=" ^ 70)
    println("Experiment complete! $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    println("=" ^ 70)
end

main()
