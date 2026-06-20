include("src/FourDInference.jl")
using .FourDInference
using Random

Random.seed!(42)

X, y, diff = generate_dataset(100; seed=42)
model = create_model!(3; hidden_dim=16, memory_neurons=10, max_steps=5, n_classes=6, seed=42)

x = X[1, :]
target = y[1]

# Compute losses for all 3 slices
model.memory.state .= 0.0
model.phase_manager.current = RETRIEVE
losses = zeros(3)
for k in 1:3
    losses[k] = compute_slice_loss_only(model, x, k, target)
end

println("Per-slice losses: ", losses)
println("Min loss: $(minimum(losses)), Max loss: $(maximum(losses))")
println()

# Now check the router scores
model.memory.state .= 0.0
model.phase_manager.current = RETRIEVE
phase_embed = encode_phase(model.phase_manager, RETRIEVE)
router_input = vcat(x, model.memory.state, phase_embed)
r = router_forward_with_cache(model.router, router_input)
router_scores = r.slice_logits[1:3]
println("Router scores: ", router_scores)

# Compute ranking gradient
grad = compute_ranking_gradient!(router_scores, losses)
println("Ranking gradient: ", grad)
println()

# Check which pairs will produce gradients
for i in 1:3
    for j in 1:3
        if i != j && losses[i] < losses[j]
            dif = router_scores[i] - router_scores[j]
            println("Pair ($i, $j): loss[$i]=$(round(losses[i], digits=3)) < loss[$j]=$(round(losses[j], digits=3)), score_diff=$dif")
        end
    end
end
