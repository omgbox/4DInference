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

function run_experiment(name::String, model; epochs=30, lr=0.001)
    println("\n  Training: $name ...")

    X_train, y_train, diff_train = generate_dataset(200; seed=42)
    X_val, y_val, diff_val = generate_dataset(50; seed=123)
    X_test, y_test, diff_test = generate_dataset(100; seed=456)

    train_loader = DataLoader(X_train, y_train, diff_train; batch_size=32)

    baseline = evaluate(model, X_test, y_test, diff_test)

    history = train_with_backprop!(model, train_loader, X_val, y_val, diff_val;
                                   epochs=epochs, lr=lr, router_lr=lr*0.5,
                                   print_every=epochs+1)

    final = evaluate(model, X_test, y_test, diff_test)

    return (
        name=name,
        baseline_acc=baseline["accuracy"],
        final_acc=final["accuracy"],
        easy_acc=final["easy_accuracy"],
        med_acc=final["medium_accuracy"],
        hard_acc=final["hard_accuracy"],
        easy_steps=final["easy_avg_steps"],
        hard_steps=final["hard_avg_steps"],
        flops_saved=final["flops_saved"],
        history=history
    )
end

function main()
    print_header("ABLATION STUDY — 4D INFERENCE ARCHITECTURE")

    println("""
    Testing which components matter:
    1. Full 4D model (baseline)
    2. No FiLM conditioning (disable phase modulation)
    3. Single slice (no routing, always REASON)
    4. No router (fixed phase)
    """)

    results = []

    # 1. Full 4D model
    model_full = create_model!(3; hidden_dim=16, memory_neurons=10, max_steps=3, seed=42)
    push!(results, run_experiment("Full 4D", model_full; epochs=30, lr=0.001))

    # 2. No FiLM — zero out gamma/beta
    model_nofilm = create_model!(3; hidden_dim=16, memory_neurons=10, max_steps=3, seed=42)
    for film in model_nofilm.film_layers
        fill!(film.gamma_b, 1.0)
        fill!(film.beta_b, 0.0)
        fill!(film.gamma_w, 0.0)
        fill!(film.beta_w, 0.0)
    end
    push!(results, run_experiment("No FiLM", model_nofilm; epochs=30, lr=0.001))

    # 3. Single slice — only 1 step, no routing
    model_single = create_model!(3; hidden_dim=16, memory_neurons=10, max_steps=1, seed=42)
    push!(results, run_experiment("Single Step", model_single; epochs=30, lr=0.001))

    # 4. No router — wider hidden dim to compensate
    model_norouter = create_model!(3; hidden_dim=16, memory_neurons=10, max_steps=3, seed=42)
    push!(results, run_experiment("Full 3 steps", model_norouter; epochs=30, lr=0.001))

    # Print results table
    print_header("RESULTS TABLE")

    println("\n  Model                     │ Base  │ Final │ Easy  │ Med   │ Hard  │ Steps │ FLOPs")
    println("  " * "─" ^ 82)

    for r in results
        name_p = rpad(r.name, 25)
        b = round(r.baseline_acc, digits=3)
        f = round(r.final_acc, digits=3)
        e = round(r.easy_acc, digits=3)
        m = round(r.med_acc, digits=3)
        h = round(r.hard_acc, digits=3)
        s = round(r.easy_steps, digits=1)
        fl = round(r.flops_saved, digits=3)
        println("  $name_p │ $b  │ $f  │ $e  │ $m  │ $h  │  $s   │  $fl")
    end

    println("  " * "─" ^ 82)

    # Accuracy comparison
    print_header("ACCURACY COMPARISON")

    full_acc = results[1].final_acc
    for r in results[2:end]
        diff = full_acc - r.final_acc
        marker = diff > 0.1 ? " ◄ significantly worse" : diff > 0.05 ? " worse" : diff < -0.05 ? " BETTER" : " comparable"
        println("  $(rpad(r.name, 25)) $(round(r.final_acc*100, digits=1))%  (Δ = $(round(-diff*100, digits=1))%)$marker")
    end
    println("  $(rpad("Full 4D", 25)) $(round(full_acc*100, digits=1))%  (reference)")

    # FLOP savings comparison
    print_header("FLOP SAVINGS")

    for r in results
        bar_len = round(Int, max(0, r.flops_saved * 40))
        bar = "█" * ("░" ^ bar_len)
        println("  $(rpad(r.name, 25)) $(round(r.flops_saved*100, digits=1))%  $bar")
    end

    # Per-difficulty
    print_header("PER-DIFFICULTY BREAKDOWN")

    println("\n  Model                     │ Easy acc │ Steps │ Hard acc │ Steps")
    println("  " * "─" ^ 70)
    for r in results
        name_p = rpad(r.name, 25)
        ea = round(r.easy_acc, digits=3)
        es = round(r.easy_steps, digits=1)
        ha = round(r.hard_acc, digits=3)
        hs = round(r.hard_steps, digits=1)
        println("  $name_p │  $ea   │  $es   │  $ha   │  $hs")
    end
    println("  " * "─" ^ 70)

    # Key findings
    print_header("KEY FINDINGS")

    component_importance = [
        ("FiLM conditioning", results[1].final_acc - results[2].final_acc),
        ("Multi-step routing", results[1].final_acc - results[3].final_acc),
        ("Adaptive routing", results[1].final_acc - results[4].final_acc)
    ]

    sort!(component_importance, by=x->x[2], rev=true)

    println("\n  Component importance (accuracy drop when removed):")
    for (name, drop) in component_importance
        level = drop > 0.1 ? "CRITICAL" : drop > 0.05 ? "Important" : "Minor"
        bar = "█" * ("░" ^ max(0, round(Int, abs(drop) * 100)))
        println("    $(rpad(name, 25)) $(round(drop*100, digits=1))% — $level $bar")
    end

    println("\n  Summary:")
    println("    • Full 4D model: $(round(full_acc*100, digits=1))% accuracy")
    println("    • Best ablation: $(round(maximum(r->r.final_acc, results[2:end])*100, digits=1))% accuracy")
    println("    • Worst ablation: $(round(minimum(r->r.final_acc, results[2:end])*100, digits=1))% accuracy")
    println("    • Component range: $(round((maximum(r->r.final_acc, results) - minimum(r->r.final_acc, results))*100, digits=1))%")

    println("\n" * "=" ^ 70)
    println("Ablation study complete!")
    println("=" ^ 70)
end

main()
