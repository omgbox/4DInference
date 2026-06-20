using Random

include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference

function print_header(title::String)
    width = 70
    println("\n" * "=" ^ width)
    println(" " ^ max(0, div(width - length(title), 2)) * title)
    println("=" ^ width)
end

function print_architecture()
    print_header("4D INFERENCE ARCHITECTURE")
    
    println("""
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        INPUT (3 features)                         │
    │                    [age, income, credit_score]                    │
    └──────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     INPUT PROJECTION (3 → 8)                      │
    │              Linear + ReLU embedding for slice processing         │
    └──────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    4D LEARNED ROUTER (MLP)                        │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
    │  │  Slice Selector  │  │  Phase Selector  │  │  Confidence Head │  │
    │  │    (4 logits)    │  │    (4 logits)    │  │   (sigmoid 0-1)  │  │
    │  └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘  │
    └───────────┼────────────────────┼────────────────────┼────────────┘
                │                    │                    │
                ▼                    ▼                    │
    ┌───────────────────┐  ┌─────────────────┐           │
    │  SLICE SELECTION  │  │ PHASE SELECTION │           │
    │  ┌───┐ ┌───┐      │  │ RETRIEVE        │           │
    │  │ 0 │ │ 1 │ ...  │  │ REASON          │           │
    │  └───┘ └───┘      │  │ PLAN            │           │
    │  RETRIEVE REASON   │  │ COMPRESS        │           │
    └────────┬──────────┘  └────────┬────────┘           │
             │                      │                    │
             └──────────┬───────────┘                    │
                        │                                │
                        ▼                                │
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     FiLM CONDITIONING                              │
    │    γ(s) * h + β(s)  where s = phase embedding (8-dim)            │
    │    Phase-dependent feature scaling and shifting                   │
    └──────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SELECTED SLICE (MLP)                            │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │ RETRIEVE: 2-layer MLP, hidden=16, fast approximate          │  │
    │  │ REASON:   3-layer MLP, hidden=16, deep systematic           │  │
    │  │ PLAN:     2-layer MLP, hidden=32, wide look-ahead           │  │
    │  │ COMPRESS: 3-layer MLP, hidden=8, narrow bottleneck          │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    └──────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SORN MEMORY (10 neurons)                       │
    │    Surprise-gated writes • STDP-like plasticity                   │
    │    Read: full state vector (10-dim)                               │
    └──────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    OUTPUT PROJECTION (8 → 3)                      │
    │              Final logits for 6-class classification              │
    └──────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    6-CLASS OUTPUT                                  │
    │     deny | review | standard | approve | premium | platinum       │
    └─────────────────────────────────────────────────────────────────────┘
    """)
end

function print_inference_trace(trace, input, label, difficulty)
    print_header("INFERENCE TRACE")
    
    println("\n  Input: ", round.(input, digits=3))
    println("  True Label: $label (difficulty: $difficulty)")
    println("  Predicted: $(argmax(trace.final_output))")
    println("  Correct: $(argmax(trace.final_output) == label)")
    println("\n  " * "─" ^ 66)
    println("  Step │ Slice       │ Phase     │ Confidence │ Memory Surprise")
    println("  " * "─" ^ 66)
    
    for (i, step) in enumerate(trace.steps)
        slice_bar = "█" * ("░" ^ (step.confidence * 20 |> round |> Int))
        slice_name = rpad(step.slice_name, 10)
        phase_name = rpad(string(step.phase), 8)
        conf = round(step.confidence, digits=3)
        surprise = round(step.memory_surprise, digits=4)
        
        println("  $(lpad(i, 4)) │ $slice_name │ $phase_name │ $slice_bar $conf │ $surprise")
    end
    
    println("  " * "─" ^ 66)
    
    println("\n  Phase History: ", join([string(p)[1:3] for p in trace.phase_history], " → "))
    println("  Slice History: ", join(trace.slice_history, " → "))
    println("  Total Steps: $(trace.total_steps)")
    
    # Show output distribution
    probs = trace.final_output
    probs = exp.(probs .- maximum(probs))
    probs = probs ./ sum(probs)
    
    println("\n  Output Distribution:")
    labels = ["deny", "review", "standard", "approve", "premium", "platinum"]
    for (i, (p, l)) in enumerate(zip(probs, labels))
        bar_len = round(p * 40) |> Int
        bar = "█" * ("░" ^ bar_len)
        marker = i == argmax(probs) ? " ◄" : ""
        println("    $(rpad(l, 10)) $(round(p, digits=3)) $bar$marker")
    end
end

function print_phase_visualization()
    print_header("PHASE TRANSITIONS")
    
    println("""
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     4 DISCRETE PHASES                             │
    │                                                                   │
    │  RETRIEVE (System 1)     REASON (System 2)                        │
    │  ┌─────────────────┐     ┌─────────────────┐                      │
    │  │ Fast, approximate│     │ Slow, systematic │                      │
    │  │ Pattern matching │     │ Rule-based logic  │                      │
    │  │ Low compute cost │     │ High compute cost │                      │
    │  └─────────────────┘     └─────────────────┘                      │
    │         ▲                         ▲                               │
    │         │                         │                               │
    │         └─────────────────────────┘                               │
    │                    bidirectional                                   │
    │         ┌─────────────────────────┐                               │
    │         │                         │                               │
    │         ▼                         ▼                               │
    │  ┌─────────────────┐     ┌─────────────────┐                      │
    │  │ Look-ahead, plan │     │ Summarize, reduce│                      │
    │  │ Future states    │     │ Memory load       │                      │
    │  │ Strategic depth  │     │ Compression       │                      │
    │  └─────────────────┘     └─────────────────┘                      │
    │  PLAN                  COMPRESS                                   │
    │                                                                   │
    │  Router learns WHEN to switch phases based on input complexity    │
    └─────────────────────────────────────────────────────────────────────┘
    """)
end

function print_slice_visualization()
    print_header("SLICE HETEROGENEITY")
    
    println("""
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    4 HETEROGENEOUS SLICES                          │
    │                                                                   │
    │  RETRIEVE          REASON           PLAN           COMPRESS       │
    │  ┌─────────┐       ┌─────────┐      ┌─────────┐    ┌─────────┐   │
    │  │ ┌─────┐ │       │ ┌─────┐ │      │ ┌─────┐ │    │ ┌─────┐ │   │
    │  │ │ 16  │ │       │ │ 16  │ │      │ │ 32  │ │    │ │  8  │ │   │
    │  │ ├─────┤ │       │ ├─────┤ │      │ ├─────┤ │    │ ├─────┤ │   │
    │  │ │ 16  │ │       │ │ 16  │ │      │ │ 32  │ │    │ │  8  │ │   │
    │  │ ├─────┤ │       │ ├─────┤ │      │ ├─────┤ │    │ ├─────┤ │   │
    │  │ │  8  │ │       │ │ 16  │ │      │ │  8  │ │    │ │  8  │ │   │
    │  │ └─────┘ │       │ ├─────┤ │      │ └─────┘ │    │ ├─────┤ │   │
    │  └─────────┘       │ │  8  │ │      └─────────┘    │ │  8  │ │   │
    │  2 layers          │ └─────┘ │                     │ └─────┘ │   │
    │  hidden=16         └─────────┘                     └─────────┘   │
    │  ~512 params       3 layers          2 layers      3 layers     │
    │                    hidden=16          hidden=32     hidden=8     │
    │                    ~768 params        ~1032 params  ~200 params │
    │                                                                   │
    │  Each slice has different: depth, width, parameter count         │
    │  Router learns to match input complexity to slice capacity       │
    └─────────────────────────────────────────────────────────────────────┘
    """)
end

function print_training_animation(model, X_train, y_train, diff_train, X_val, y_val, diff_val)
    print_header("LIVE TRAINING TRACE")
    
    train_loader = DataLoader(X_train, y_train, diff_train; batch_size=32)
    
    println("\n  Training for 30 epochs (watch the routing evolve)...")
    println("\n  " * "─" ^ 66)
    println("  Epoch │ Loss    │ Accuracy │ Easy Steps │ Hard Steps │ FLOPs Saved")
    println("  " * "─" ^ 66)
    
    for epoch in 1:30
        # Train one epoch
        epoch_loss = 0.0
        n_batches = 0
        for _ in 1:length(train_loader)
            X_batch, y_batch, diff_batch = next_batch!(train_loader)
            total_loss, _, _, _, _, _ = compute_loss(model, X_batch, y_batch, diff_batch)
            perturb_and_update!(model, X_batch, y_batch, diff_batch, 0.001)
            epoch_loss += total_loss
            n_batches += 1
        end
        epoch_loss /= n_batches
        
        # Evaluate
        metrics = evaluate(model, X_val, y_val, diff_val)
        
        # Visual progress bar
        acc_bar = round(metrics["accuracy"] * 20) |> Int
        acc_visual = "█" * ("░" ^ acc_bar) * (" " ^ (20 - acc_bar))
        
        loss_visual = "░" ^ (20 - min(20, round(epoch_loss * 10) |> Int))
        
        println("  $(lpad(epoch, 5)) │ $(round(epoch_loss, digits=3))  │ $(round(metrics["accuracy"], digits=3))   │    $(round(metrics["easy_avg_steps"], digits=1))     │    $(round(metrics["hard_avg_steps"], digits=1))     │   $(round(metrics["flops_saved"], digits=3))")
    end
    
    println("  " * "─" ^ 66)
    
    # Show final routing distribution
    println("\n  Final Routing Distribution (on validation set):")
    sample_traces = []
    for i in 1:min(100, size(X_val, 1))
        trace = forward(model, @view X_val[i, :])
        push!(sample_traces, trace)
    end
    
    slice_counts = zeros(4)
    phase_counts = zeros(4)
    for trace in sample_traces
        for s in trace.slice_history
            slice_counts[s] += 1
        end
        for p in trace.phase_history
            phase_counts[Int(p)] += 1
        end
    end
    
    slice_labels = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]
    phase_labels = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]
    
    println("\n  Slice Usage:")
    for (i, (count, label)) in enumerate(zip(slice_counts, slice_labels))
        bar_len = round(count / sum(slice_counts) * 30) |> Int
        bar = "█" * ("░" ^ bar_len)
        println("    $(rpad(label, 10)) $(round(count/sum(slice_counts)*100, digits=1))% $bar")
    end
    
    println("\n  Phase Usage:")
    for (i, (count, label)) in enumerate(zip(phase_counts, phase_labels))
        bar_len = round(count / sum(phase_counts) * 30) |> Int
        bar = "█" * ("░" ^ bar_len)
        println("    $(rpad(label, 10)) $(round(count/sum(phase_counts)*100, digits=1))% $bar")
    end
end

function print_memory_visualization(model)
    print_header("SORN MEMORY DYNAMICS")
    
    println("""
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SORN MEMORY (10 neurons)                       │
    │                                                                   │
    │  State: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]     │
    │                                                                   │
    │  Write Mechanism:                                                 │
    │  ┌──────────────────────────────────────────────────────────────┐ │
    │  │  state[i] = 0.9 * state[i] + 0.1 * input[i]  (if i <= dim) │ │
    │  │  Surprise = ||input - state||₂  (surprise threshold: 0.3)   │ │
    │  │  If surprise > threshold: STDP-like plasticity update       │ │
    │  └──────────────────────────────────────────────────────────────┘ │
    │                                                                   │
    │  Read: Returns full 10-dim state vector                          │
    │  Connected to router input for conditioning                      │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    # Show memory evolution during inference
    mem = model.memory
    old_state = copy(mem.state)
    
    println("  Memory evolution during inference:")
    println("  " * "─" ^ 66)
    println("  Step │ Neuron States (bar = |state|)")
    println("  " * "─" ^ 66)
    
    test_input = randn(3)
    println("  0    │ $(round.(mem.state, digits=3))")
    
    for step in 1:5
        trace = forward(model, test_input)
        state_after = copy(mem.state)
        bar = [round(abs(s) * 20) |> Int for s in state_after]
        bar_parts = [rpad("█" * ("░" ^ max(0, 5-b)), 6) for b in bar[1:5]]
        bar_str = join(bar_parts)
        println("  $step   │ $bar_str")
    end
    
    println("  " * "─" ^ 66)
end

function main()
    println("\n" * "█" ^ 70)
    println("█" * " " ^ 68 * "█")
    println("█" * "      4D INFERENCE ARCHITECTURE — VISUALIZATION" * " " ^ 22 * "█")
    println("█" * "      Pure Julia • CPU-only • Learned Routing" * " " ^ 24 * "█")
    println("█" * " " ^ 68 * "█")
    println("█" ^ 70)
    
    # 1. Architecture diagram
    print_architecture()
    
    # 2. Phase visualization
    print_phase_visualization()
    
    # 3. Slice visualization
    print_slice_visualization()
    
    # 4. Memory visualization
    model = create_model!(3; hidden_dim=8, memory_neurons=5, max_steps=3, seed=42)
    print_memory_visualization(model)
    
    # 5. Generate data and run training trace
    X_train, y_train, diff_train = generate_dataset(200; seed=42)
    X_val, y_val, diff_val = generate_dataset(50; seed=123)
    
    # 6. Live training animation
    print_training_animation(model, X_train, y_train, diff_train, X_val, y_val, diff_val)
    
    # 7. Sample inference traces
    print_header("SAMPLE INFERENCE TRACES")
    
    for i in 1:3
        idx = rand(1:size(X_val, 1))
        input = @view X_val[idx, :]
        label = y_val[idx]
        diff = diff_val[idx]
        
        trace = forward(model, input)
        print_inference_trace(trace, input, label, diff)
        println()
    end
    
    println("\n" * "=" ^ 70)
    println("Visualization complete!")
    println("=" ^ 70)
end

main()
