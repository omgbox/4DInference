using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference

println("=" ^ 70)
println("4D Inference — Backprop Training Test")
println("=" ^ 70)

# Generate data
println("\nGenerating dataset...")
X_train, y_train, diff_train = generate_dataset(300; seed=42)
X_val, y_val, diff_val = generate_dataset(100; seed=123)
X_test, y_test, diff_test = generate_dataset(200; seed=456)

println("Training: $(size(X_train, 1)) | Val: $(size(X_val, 1)) | Test: $(size(X_test, 1))")

# Create data loaders
train_loader = DataLoader(X_train, y_train, diff_train; batch_size=32)

# Create model
println("\nCreating model...")
model = create_model!(3; hidden_dim=16, memory_neurons=10, max_steps=5, seed=42)

# Count parameters
let tp = 0
    for slice in model.slices
        for W in slice.weights
            tp += length(W)
        end
        for b in slice.biases
            tp += length(b)
        end
    end
    tp += length(model.output_w) + length(model.output_b)
    tp += length(model.input_proj_w) + length(model.input_proj_b)
    println("Model parameters: $tp")
end

# Baseline evaluation (untrained)
println("\n" * "-" ^ 70)
println("Untrained baseline:")
baseline_metrics = evaluate(model, X_test, y_test, diff_test)
println("  Accuracy: $(round(baseline_metrics["accuracy"], digits=3))")
println("  Easy acc: $(round(baseline_metrics["easy_accuracy"], digits=3))")
println("  Hard acc: $(round(baseline_metrics["hard_accuracy"], digits=3))")
println("  FLOPs saved: $(round(baseline_metrics["flops_saved"], digits=3))")

# Train with backprop
println("\n" * "-" ^ 70)
println("Training with backpropagation + REINFORCE...")
println("-" ^ 70)

history = train_with_backprop!(model, train_loader, X_val, y_val, diff_val;
                               epochs=50, lr=0.005, router_lr=0.001,
                               entropy_coeff=0.05, print_every=5)

# Final evaluation
println("\n" * "=" ^ 70)
println("Final Results")
println("=" ^ 70)

# Test set evaluation
test_metrics = evaluate(model, X_test, y_test, diff_test)
println("\nTest Set:")
println("  Overall accuracy: $(round(test_metrics["accuracy"], digits=3))")
println("  Easy accuracy: $(round(test_metrics["easy_accuracy"], digits=3))")
println("  Medium accuracy: $(round(test_metrics["medium_accuracy"], digits=3))")
println("  Hard accuracy: $(round(test_metrics["hard_accuracy"], digits=3))")
println("  Easy avg steps: $(round(test_metrics["easy_avg_steps"], digits=2))")
println("  Hard avg steps: $(round(test_metrics["hard_avg_steps"], digits=2))")
println("  FLOPs saved: $(round(test_metrics["flops_saved"], digits=3))")

# Improvement over baseline
improvement = test_metrics["accuracy"] - baseline_metrics["accuracy"]
println("\n  Improvement over baseline: $(round(improvement * 100, digits=1))%")

# Detailed trace for hard sample
println("\n" * "-" ^ 70)
println("Sample trace (hard sample):")
hard_indices = findall(diff_test .== 3)
if length(hard_indices) > 0
    idx = hard_indices[1]
    input = @view X_test[idx, :]
    label = y_test[idx]

    println("  Input: $(round.(input, digits=3))")
    println("  True label: $label")

    trace = forward(model, input; verbose=true)
    println("  Predicted: $(argmax(trace.final_output))")
    println("  Correct: $(argmax(trace.final_output) == label)")
    println("  Phase history: $(trace.phase_history)")
    println("  Slice history: $(trace.slice_history)")
end

# Slice distribution analysis
println("\n" * "-" ^ 70)
println("Routing analysis (100 samples):")
slice_counts = zeros(4)
phase_counts = zeros(4)
step_counts = Float64[]

for i in 1:min(100, size(X_test, 1))
    trace = forward(model, @view X_test[i, :])
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

println("\n  Steps per sample:")
println("    Mean: $(round(mean(step_counts), digits=2))")
println("    Min: $(round(minimum(step_counts), digits=0))")
println("    Max: $(round(maximum(step_counts), digits=0))")

println("\n" * "=" ^ 70)
println("Backprop training test complete!")
println("=" ^ 70)
