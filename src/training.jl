export train!, evaluate, compute_loss, perturb_and_update!

using Statistics
using Random

function _softmax_loss(x::AbstractVector{Float64})
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    exp_x ./ sum(exp_x)
end

function cross_entropy_loss(pred::AbstractVector{Float64}, target::Int)
    probs = _softmax_loss(pred)
    target_idx = clamp(target, 1, length(probs))
    return -log(max(probs[target_idx], 1e-8))
end

function cross_entropy_loss_batch(preds::AbstractMatrix{Float64}, targets::AbstractVector{Int})
    batch_size = length(targets)
    total_loss = 0.0
    @simd for i in 1:batch_size
        @inbounds total_loss += cross_entropy_loss(@view(preds[i, :]), targets[i])
    end
    return total_loss / batch_size
end

function difficulty_loss(traces::Vector, difficulties::AbstractVector{Int})
    total = 0.0
    @simd for i in eachindex(traces)
        @inbounds begin
            expected_steps = difficulties[i]
            actual_steps = traces[i].total_steps
            total += (actual_steps - expected_steps)^2
        end
    end
    return total / length(traces)
end

function _entropy_loss(traces::Vector)
    total = 0.0
    for trace in traces
        if length(trace.slice_history) > 0
            slice_counts = zeros(N_ROUTED_SLICES)
            for s in trace.slice_history
                slice_counts[s] += 1.0
            end
            slice_probs = slice_counts ./ sum(slice_counts)
            entropy = 0.0
            for p in slice_probs
                if p > 0
                    entropy -= p * log(p)
                end
            end
            total += entropy
        end
    end
    return -total / length(traces)
end

function phase_diversity_loss(traces::Vector)
    total = 0.0
    all_phases = PhaseType[]
    for trace in traces
        append!(all_phases, trace.phase_history)
    end

    if length(all_phases) > 0
        phase_counts = zeros(4)
        for p in all_phases
            phase_counts[Int(p)] += 1.0
        end
        phase_probs = phase_counts ./ sum(phase_counts)
        entropy = 0.0
        for p in phase_probs
            if p > 0
                entropy -= p * log(p)
            end
        end
        total = entropy
    end

    return -total
end

function compute_loss(model::FourDModel, X_batch::AbstractMatrix{Float64},
                      y_batch::AbstractVector{Int},
                      difficulty_batch::AbstractVector{Int};
                      λ_diff::Float64=0.1,
                      λ_entropy::Float64=0.01,
                      λ_phase::Float64=0.01)
    preds, traces = forward(model, X_batch)

    L_task = cross_entropy_loss_batch(preds, y_batch)
    L_diff = difficulty_loss(traces, difficulty_batch)
    L_entropy = _entropy_loss(traces)
    L_phase = phase_diversity_loss(traces)

    total = L_task + λ_diff * L_diff + λ_entropy * L_entropy + λ_phase * L_phase

    return total, L_task, L_diff, L_entropy, L_phase, traces
end

function evaluate(model::FourDModel, X::AbstractMatrix{Float64},
                  y::AbstractVector{Int},
                  difficulty::AbstractVector{Int})
    preds, traces = forward(model, X)

    pred_labels = [argmax(@view preds[i, :]) for i in 1:size(preds, 1)]
    accuracy = sum(pred_labels .== y) / length(y)

    easy_mask = difficulty .== 1
    med_mask = difficulty .== 2
    hard_mask = difficulty .== 3

    easy_acc = sum(easy_mask) > 0 ? sum(pred_labels[easy_mask] .== y[easy_mask]) / sum(easy_mask) : 0.0
    med_acc = sum(med_mask) > 0 ? sum(pred_labels[med_mask] .== y[med_mask]) / sum(med_mask) : 0.0
    hard_acc = sum(hard_mask) > 0 ? sum(pred_labels[hard_mask] .== y[hard_mask]) / sum(hard_mask) : 0.0

    easy_steps = sum(easy_mask) > 0 ? mean([traces[i].total_steps for i in 1:length(traces) if easy_mask[i]]) : 0.0
    med_steps = sum(med_mask) > 0 ? mean([traces[i].total_steps for i in 1:length(traces) if med_mask[i]]) : 0.0
    hard_steps = sum(hard_mask) > 0 ? mean([traces[i].total_steps for i in 1:length(traces) if hard_mask[i]]) : 0.0

    total_flops = 0.0
    for trace in traces
        for step in trace.steps
            total_flops += step.slice_idx == 2 ? 1.0 : 0.5
        end
        total_flops += trace.total_steps * 0.5
    end
    baseline_flops = length(traces) * model.max_steps * 2.0
    flops_saved = 1.0 - (total_flops / baseline_flops)

    return Dict(
        "accuracy" => accuracy,
        "easy_accuracy" => easy_acc,
        "medium_accuracy" => med_acc,
        "hard_accuracy" => hard_acc,
        "easy_avg_steps" => easy_steps,
        "medium_avg_steps" => med_steps,
        "hard_avg_steps" => hard_steps,
        "flops_saved" => flops_saved,
        "total_samples" => length(y)
    )
end

function train!(model::FourDModel, train_data::DataLoader, val_X::AbstractMatrix{Float64},
                val_y::AbstractVector{Int}, val_diff::AbstractVector{Int};
                epochs::Int=100, lr::Float64=0.001, print_every::Int=10)
    history = Dict{String, Vector{Float64}}(
        "loss" => Float64[],
        "accuracy" => Float64[],
        "easy_steps" => Float64[],
        "hard_steps" => Float64[]
    )

    for epoch in 1:epochs
        epoch_loss = 0.0
        n_batches = 0

        for _ in 1:length(train_data)
            X_batch, y_batch, diff_batch = next_batch!(train_data)

            total_loss, L_task, L_diff, L_entropy, L_phase, _ = compute_loss(
                model, X_batch, y_batch, diff_batch
            )

            perturb_and_update!(model, X_batch, y_batch, diff_batch, lr)

            epoch_loss += total_loss
            n_batches += 1
        end

        epoch_loss /= n_batches
        push!(history["loss"], epoch_loss)

        if epoch % print_every == 0 || epoch == 1
            metrics = evaluate(model, val_X, val_y, val_diff)
            push!(history["accuracy"], metrics["accuracy"])
            push!(history["easy_steps"], metrics["easy_avg_steps"])
            push!(history["hard_steps"], metrics["hard_avg_steps"])

            println("Epoch $epoch: loss=$(round(epoch_loss, digits=4)), " *
                    "acc=$(round(metrics["accuracy"], digits=3)), " *
                    "easy_steps=$(round(metrics["easy_avg_steps"], digits=2)), " *
                    "hard_steps=$(round(metrics["hard_avg_steps"], digits=2)), " *
                    "flops_saved=$(round(metrics["flops_saved"], digits=3))")
        end
    end

    return history
end

function perturb_and_update!(model::FourDModel, X_batch, y_batch, diff_batch, lr::Float64)
    current_loss, _, _, _, _, _ = compute_loss(model, X_batch, y_batch, diff_batch)

    scale = 0.01

    for i in eachindex(model.output_w)
        old_val = model.output_w[i]
        model.output_w[i] += randn() * scale
        new_loss, _, _, _, _, _ = compute_loss(model, X_batch, y_batch, diff_batch)
        if new_loss < current_loss
            current_loss = new_loss
        else
            model.output_w[i] = old_val
        end
    end

    for i in eachindex(model.output_b)
        old_val = model.output_b[i]
        model.output_b[i] += randn() * scale
        new_loss, _, _, _, _, _ = compute_loss(model, X_batch, y_batch, diff_batch)
        if new_loss < current_loss
            current_loss = new_loss
        else
            model.output_b[i] = old_val
        end
    end

    for slice in model.slices
        for W in slice.weights
            n_perturb = max(1, div(length(W), 10))
            indices = randperm(length(W))[1:n_perturb]
            for idx in indices
                old_val = W[idx]
                W[idx] += randn() * scale * 0.1
                new_loss, _, _, _, _, _ = compute_loss(model, X_batch, y_batch, diff_batch)
                if new_loss < current_loss
                    current_loss = new_loss
                else
                    W[idx] = old_val
                end
            end
        end
    end
end
