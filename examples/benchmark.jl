using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference

function print_header(title::String)
    width = 70
    println("\n" * "=" ^ width)
    println(" " ^ max(0, div(width - length(title), 2)) * title)
    println("=" ^ width)
end

function run_benchmark(name::String, model; epochs=30, lr=0.001)
    println("\n  Benchmarking: $name ...")

    X_train, y_train, diff_train = generate_dataset(300; seed=42)
    X_val, y_val, diff_val = generate_dataset(100; seed=123)
    X_test, y_test, diff_test = generate_dataset(200; seed=456)

    train_loader = DataLoader(X_train, y_train, diff_train; batch_size=32)

    params = count_parameters(model)

    t_start = time()
    history = train_with_backprop!(model, train_loader, X_val, y_val, diff_val;
                                   epochs=epochs, lr=lr, router_lr=lr*0.5,
                                   print_every=epochs+1)
    t_train = time() - t_start

    t_start = time()
    final = evaluate(model, X_test, y_test, diff_test)
    t_infer = time() - t_start / length(y_test)

    return (
        name=name,
        params=params,
        accuracy=final["accuracy"],
        easy_acc=final["easy_accuracy"],
        med_acc=final["medium_accuracy"],
        hard_acc=final["hard_accuracy"],
        easy_steps=final["easy_avg_steps"],
        hard_steps=final["hard_avg_steps"],
        flops_saved=final["flops_saved"],
        train_time=t_train,
        history=history
    )
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
    print_header("BENCHMARK — 4D vs SCALED MODELS")

    results = []

    # 1. Small 4D (current)
    model_small = create_model!(3; hidden_dim=16, memory_neurons=10, max_steps=3, seed=42)
    push!(results, run_benchmark("4D Small (16h, 10m)", model_small; epochs=30, lr=0.001))

    # 2. Medium 4D
    model_med = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=3, seed=42)
    push!(results, run_benchmark("4D Medium (32h, 50m)", model_med; epochs=30, lr=0.0008))

    # 3. Large 4D
    model_large = create_model!(3; hidden_dim=64, memory_neurons=100, max_steps=3, seed=42)
    push!(results, run_benchmark("4D Large (64h, 100m)", model_large; epochs=30, lr=0.0005))

    # 4. Single slice baseline (equivalent to plain MLP)
    model_single = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=1, seed=42)
    push!(results, run_benchmark("MLP Baseline (32h)", model_single; epochs=30, lr=0.001))

    # 5. No routing (fixed 3 steps)
    model_noroute = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=3, seed=42)
    push!(results, run_benchmark("Fixed 3-Step (32h, 50m)", model_noroute; epochs=30, lr=0.0008))

    # Print results
    print_header("BENCHMARK RESULTS")

    println("\n  Model                     │ Params │ Acc    │ Easy   │ Hard   │ Steps │ FLOPs  │ Time")
    println("  " * "─" ^ 90)

    for r in results
        name_p = rpad(r.name, 25)
        p = rpad(string(r.params), 6)
        a = round(r.accuracy, digits=3)
        e = round(r.easy_acc, digits=3)
        h = round(r.hard_acc, digits=3)
        s = round(r.easy_steps, digits=1)
        f = round(r.flops_saved, digits=3)
        t = round(r.train_time, digits=1)
        println("  $name_p │ $p │ $a  │ $e  │ $h  │  $s   │  $f   │  $(t)s")
    end

    println("  " * "─" ^ 90)

    # Accuracy vs Parameters analysis
    print_header("ACCURACY vs PARAMETERS")

    println("\n  Efficiency = Accuracy / Parameters")
    println("\n  Model                     │ Params │ Acc    │ Efficiency")
    println("  " * "─" ^ 70)

    for r in results
        name_p = rpad(r.name, 25)
        p = r.params
        a = r.accuracy
        eff = a / p * 1000
        bar_len = round(Int, eff * 20)
        bar = "█" * ("░" ^ max(0, bar_len))
        println("  $name_p │ $p │ $a  │ $(round(eff, digits=4)) $bar")
    end

    println("  " * "─" ^ 70)

    # FLOP savings comparison
    print_header("FLOP SAVINGS COMPARISON")

    for r in results
        bar_len = round(Int, max(0, r.flops_saved * 50))
        bar = "█" * ("░" ^ bar_len)
        println("  $(rpad(r.name, 25)) $(round(r.flops_saved*100, digits=1))%  $bar")
    end

    # Per-difficulty breakdown
    print_header("PER-DIFFICULTY BREAKDOWN")

    println("\n  Model                     │ Easy acc │ Hard acc │ Gap     │ Adaptive?")
    println("  " * "─" ^ 75)

    for r in results
        name_p = rpad(r.name, 25)
        ea = round(r.easy_acc, digits=3)
        ha = round(r.hard_acc, digits=3)
        gap = round(r.easy_acc - r.hard_acc, digits=3)
        adaptive = abs(gap) > 0.1 ? "YES" : "no"
        println("  $name_p │  $ea   │  $ha   │  $gap   │ $adaptive")
    end

    println("  " * "─" ^ 75)

    # Key findings
    print_header("KEY FINDINGS")

    best_acc = maximum(r -> r.accuracy, results)
    _, best_idx = findmax(r -> r.accuracy, results)
    best_model = results[best_idx]

    _, most_efficient_idx = findmax(r -> r.accuracy / r.params, results)
    most_efficient = results[most_efficient_idx]

    println("""
    1. Best accuracy: $(round(best_acc*100, digits=1))% — $(best_model.name)
    2. Most efficient: $(most_efficient.name)
       $(most_efficient.params) params, $(round(most_efficient.accuracy*100, digits=1))% accuracy

    3. Scaling effects:""")

    for r in results
        if r.accuracy > 0.5
            println("    ✓ $(r.name): $(round(r.accuracy*100, digits=1))% accuracy")
        else
            println("    ✗ $(r.name): $(round(r.accuracy*100, digits=1))% accuracy")
        end
    end

    # Scaling trend
    if length(results) >= 3
        accs = [r.accuracy for r in results[1:3]]
        params = [r.params for r in results[1:3]]

        println("\n  4. Scaling trend (small → medium → large):")
        for i in 1:length(accs)-1
            p_ratio = params[i+1] / params[i]
            a_diff = accs[i+1] - accs[i]
            println("    $(params[i]) → $(params[i+1]) params ($(round(p_ratio, digits=1))x)")
            println("      Accuracy: $(round(accs[i]*100, digits=1))% → $(round(accs[i+1]*100, digits=1))% ($(round(a_diff*100, digits=1))%)")
        end
    end

    # Routing analysis for large model
    print_header("ROUTING ANALYSIS — LARGE MODEL")

    X_test, y_test, diff_test = generate_dataset(200; seed=456)

    slice_counts = zeros(4)
    phase_counts = zeros(4)
    step_counts = Float64[]

    for i in 1:min(100, size(X_test, 1))
        trace = forward(model_large, @view X_test[i, :])
        for s in trace.slice_history
            slice_counts[s] += 1
        end
        for p in trace.phase_history
            phase_counts[Int(p)] += 1
        end
        push!(step_counts, trace.total_steps)
    end

    slice_labels = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]
    phase_labels = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]

    println("\n  Slice Usage:")
    for (count, label) in zip(slice_counts, slice_labels)
        pct = count / sum(slice_counts) * 100
        bar = "█" * ("░" ^ round(Int, pct / 3))
        println("    $(rpad(label, 10)) $(round(pct, digits=1))% $bar")
    end

    println("\n  Phase Usage:")
    for (count, label) in zip(phase_counts, phase_labels)
        pct = count / sum(phase_counts) * 100
        bar = "█" * ("░" ^ round(Int, pct / 3))
        println("    $(rpad(label, 10)) $(round(pct, digits=1))% $bar")
    end

    println("\n  Steps per sample: $(round(mean(step_counts), digits=2)) mean")

    println("\n" * "=" ^ 70)
    println("Benchmark complete!")
    println("=" ^ 70)
end

main()
