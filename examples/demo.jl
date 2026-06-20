using Statistics
using Printf

include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference

println("=" ^ 60)
println("4D Inference Architecture - Demo")
println("=" ^ 60)

# Generate dataset
println("\nGenerating dataset...")
X_train, y_train, diff_train = generate_dataset(500; seed=42)
X_val, y_val, diff_val = generate_dataset(100; seed=123)
X_test, y_test, diff_test = generate_dataset(200; seed=456)

println("Training set: $(size(X_train, 1)) samples")
println("Validation set: $(size(X_val, 1)) samples")
println("Test set: $(size(X_test, 1)) samples")

# Create data loaders
train_loader = DataLoader(X_train, y_train, diff_train; batch_size=32)
val_loader = DataLoader(X_val, y_val, diff_val; batch_size=32)

# Create model
println("\nCreating model...")
model = create_model!(3; hidden_dim=16, memory_neurons=10, max_steps=5, seed=42)
println("Model created")

# Train
println("\nTraining...")
history = train!(model, train_loader, X_val, y_val, diff_val;
                 epochs=50, lr=0.001, print_every=10)

# Final evaluation
println("\n" * "=" ^ 60)
println("Final Evaluation")
println("=" ^ 60)

# Easy test
X_easy, y_easy, diff_easy = generate_dataset(200; seed=789)
easy_metrics = evaluate(model, X_easy, y_easy, diff_easy)
println("\nEasy Test:")
println("  Accuracy: $(round(easy_metrics["accuracy"], digits=3))")
println("  Avg steps: $(round(easy_metrics["easy_avg_steps"], digits=2))")

# Medium test
X_med, y_med, diff_med = generate_dataset(200; seed=101)
med_metrics = evaluate(model, X_med, y_med, diff_med)
println("\nMedium Test:")
println("  Accuracy: $(round(med_metrics["accuracy"], digits=3))")
println("  Avg steps: $(round(med_metrics["medium_avg_steps"], digits=2))")

# Hard test
hard_metrics = evaluate(model, X_test, y_test, diff_test)
println("\nHard Test:")
println("  Accuracy: $(round(hard_metrics["accuracy"], digits=3))")
println("  Avg steps: $(round(hard_metrics["hard_avg_steps"], digits=2))")

# FLOP savings
println("\nFLOP Savings: $(round(hard_metrics["flops_saved"], digits=3))")

# Detailed trace for a single sample
println("\n" * "=" ^ 60)
println("Sample Inference Trace")
println("=" ^ 60)

sample_idx = 1
sample_input = @view X_test[sample_idx, :]
sample_label = y_test[sample_idx]
sample_diff = diff_test[sample_idx]

println("\nInput: $(round.(sample_input, digits=3))")
println("True label: $sample_label (difficulty: $sample_diff)")

trace = forward(model, sample_input; verbose=true)

println("\nFinal output: $(round.(trace.final_output, digits=3))")
println("Predicted label: $(argmax(trace.final_output))")
println("Correct: $(argmax(trace.final_output) == sample_label)")

println("\nPhase history: $(trace.phase_history)")
println("Slice history: $(trace.slice_history)")
println("Total steps: $(trace.total_steps)")

println("\nResults Summary")
println("Easy accuracy: $(round(easy_metrics["accuracy"], digits=3))")
println("Medium accuracy: $(round(med_metrics["accuracy"], digits=3))")
println("Hard accuracy: $(round(hard_metrics["accuracy"], digits=3))")
println("FLOP savings: $(round(hard_metrics["flops_saved"], digits=3))")

println("\nDemo complete!")
