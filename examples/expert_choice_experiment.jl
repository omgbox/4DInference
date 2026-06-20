using Random
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference

function generate_multimodal_task(n_samples::Int; seq_len::Int=8, seed::Int=42)
    rng = MersenneTwister(seed)
    X = Array{Float64}(undef, n_samples, seq_len, 3)
    y = Vector{Int}(undef, n_samples)
    for i in 1:n_samples
        modality = rand(rng, 1:4)
        subclass = rand(rng, 1:3)
        if modality == 1
            slope = subclass == 1 ? 0.06 : (subclass == 2 ? -0.06 : 0.0)
            for t in 1:seq_len
                X[i, t, 1] = clamp(0.5 + slope * (t - 1) + randn(rng) * 0.03, 0.0, 1.0)
                X[i, t, 2] = clamp(0.5 + randn(rng) * 0.05, 0.0, 1.0)
                X[i, t, 3] = clamp(0.5 + randn(rng) * 0.05, 0.0, 1.0)
            end
            y[i] = subclass
        elseif modality == 2
            level = subclass == 1 ? 0.2 : (subclass == 2 ? 0.5 : 0.8)
            for t in 1:seq_len
                X[i, t, 1] = clamp(level + randn(rng) * 0.02, 0.0, 1.0)
                X[i, t, 2] = clamp(level + randn(rng) * 0.02, 0.0, 1.0)
                X[i, t, 3] = clamp(level + randn(rng) * 0.02, 0.0, 1.0)
            end
            y[i] = 3 + subclass
        elseif modality == 3
            phase_offset = subclass * 2π / 3
            for t in 1:seq_len
                v = 0.5 + 0.3 * sin(2π * t / seq_len + phase_offset)
                X[i, t, 1] = clamp(v + randn(rng) * 0.02, 0.0, 1.0)
                X[i, t, 2] = clamp(v + randn(rng) * 0.02, 0.0, 1.0)
                X[i, t, 3] = clamp(v + randn(rng) * 0.02, 0.0, 1.0)
            end
            y[i] = 6 + subclass
        else
            hot_ch = rand(rng, 1:3)
            for t in 1:seq_len
                for c in 1:3
                    if c == hot_ch
                        X[i, t, c] = clamp(0.8 + randn(rng) * 0.03, 0.0, 1.0)
                    else
                        X[i, t, c] = clamp(0.2 + randn(rng) * 0.03, 0.0, 1.0)
                    end
                end
            end
            y[i] = 9 + subclass
        end
    end
    return X, y
end

function cosine_anneal_lr(base_lr::Float64, epoch::Int, max_epochs::Int)
    return base_lr * 0.5 * (1.0 + cos(π * epoch / max_epochs))
end

function softmax_stable(x::AbstractVector{Float64})
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    s = sum(exp_x)
    if s == 0.0 || !isfinite(s)
        return ones(length(x)) ./ length(x)
    end
    exp_x ./ s
end

function main()
    println("=" ^ 70)
    println("  SOFT EXPERT CHOICE ROUTING EXPERIMENT")
    println("=" ^ 70)
    println("  Each timestep: run ALL 4 slices, combine with affinity weights")
    println("  Affinities learned via selection_head, differentiable")
    println()

    SEQ_LEN = 8
    N_TRAIN = 2000
    N_VAL = 500
    EPOCHS = 60
    BATCH_SIZE = 32

    X_train, y_train = generate_multimodal_task(N_TRAIN; seed=42)
    X_val, y_val = generate_multimodal_task(N_VAL; seed=123)

    println("  4 modalities x 3 subclasses = 12 classes")
    println("  Training: $(size(X_train)), Val: $(size(X_val))")

    model = create_model!(3; hidden_dim=48, memory_neurons=60, max_steps=3, n_classes=12, seed=42)

    rng = MersenneTwister(42)
    best_val = 0.0
    patience = 0

    for epoch in 1:EPOCHS
        lr = cosine_anneal_lr(0.002, epoch, EPOCHS)
        indices = shuffle(rng, 1:N_TRAIN)
        epoch_loss = 0.0
        n_b = 0
        slice_selection_counts = zeros(4)

        for start_idx in 1:BATCH_SIZE:N_TRAIN
            end_idx = min(start_idx + BATCH_SIZE - 1, N_TRAIN)
            batch_idx = indices[start_idx:end_idx]
            X_batch = X_train[batch_idx, :, :]
            y_batch = y_train[batch_idx]
            bs = size(X_batch, 1)

            accumulated_state = zeros(bs, model.slice_output_dim)

            for t in 1:SEQ_LEN
                x_t_batch = zeros(bs, 3)
                for b_idx in 1:bs
                    x_t_batch[b_idx, :] .= @view X_batch[b_idx, t, :]
                end

                ec_affinities = model.router.ec_selection_w * x_t_batch' .+ model.router.ec_selection_b

                ec_weights = zeros(N_SLICES, bs)
                for b_idx in 1:bs
                    ec_weights[:, b_idx] .= softmax_stable(@view ec_affinities[:, b_idx])
                end

                for b_idx in 1:bs
                    weights = @view ec_weights[:, b_idx]
                    dominant = argmax(weights)
                    slice_selection_counts[dominant] += 1.0
                end

                slice_raw_outputs = Vector{Vector{Float64}}(undef, N_SLICES)
                slice_caches_s = Vector{NamedTuple}(undef, N_SLICES)
                slice_films_s = Vector{NamedTuple}(undef, N_SLICES)

                for s in 1:N_SLICES
                    projected = model.slice_proj_ws[s] * (@view x_t_batch[1, :]) .+ model.slice_proj_bs[s]
                    slice = model.slices[s]
                    raw_out, acts, pre_acts = slice_forward_cached(slice, projected)

                    mem_state = read_memory(model.memory)
                    phase_embed = encode_phase(model.phase_manager, RETRIEVE)

                    film = model.film_layers[s]
                    gamma = film.gamma_w * phase_embed .+ film.gamma_b
                    beta = film.beta_w * phase_embed .+ film.beta_b
                    conditioned = gamma .* raw_out .+ beta

                    slice_raw_outputs[s] = conditioned
                    slice_caches_s[s] = (acts=acts, pre_acts=pre_acts, projected=projected)
                    slice_films_s[s] = (gamma=gamma, beta=beta, phase_embed=phase_embed)
                end

                for b_idx in 1:bs
                    weights = @view ec_weights[:, b_idx]
                    combined = zeros(model.slice_output_dim)
                    for s in 1:N_SLICES
                        combined .+= weights[s] .* slice_raw_outputs[s]
                    end
                    accumulated_state[b_idx, :] .+= combined
                end

                mem_state = read_memory(model.memory)
                phase_embed = encode_phase(model.phase_manager, RETRIEVE)
                update_phase!(model.phase_manager, COMPRESS)
                for s in 1:N_SLICES
                    surprise_write!(model.memory, slice_raw_outputs[s])
                end
            end

            accumulated_state ./= SEQ_LEN

            batch_loss = 0.0
            for b_idx in 1:bs
                logits = model.output_w * (@view accumulated_state[b_idx, :]) .+ model.output_b
                probs = softmax_stable(logits)
                target = y_batch[b_idx]
                loss = -log(max(probs[target], 1e-8))
                batch_loss += loss

                d_logits = copy(probs)
                d_logits[target] -= 1.0
                d_logits = clamp.(d_logits, -2.0, 2.0)

                d_avg = model.output_w' * d_logits
                d_avg = clamp.(d_avg, -2.0, 2.0)

                d_out_w = d_logits * (@view accumulated_state[b_idx, :])'
                d_out_w = clamp.(d_out_w, -1.0, 1.0)
                model.output_w .-= lr .* d_out_w
                model.output_b .-= lr .* clamp.(d_logits, -1.0, 1.0)
            end

            epoch_loss += batch_loss / bs

            d_avg_batch = zeros(model.slice_output_dim)
            for b_idx in 1:bs
                logits = model.output_w * (@view accumulated_state[b_idx, :]) .+ model.output_b
                probs = softmax_stable(logits)
                target = y_batch[b_idx]
                d_l = copy(probs)
                d_l[target] -= 1.0
                d_avg_batch .+= clamp.(model.output_w' * clamp.(d_l, -2.0, 2.0), -2.0, 2.0)
            end
            d_avg_batch ./= bs

            ec_affinities = model.router.ec_selection_w * (zeros(bs, 3)') .+ model.router.ec_selection_b
            for b_idx in 1:bs
                x_b = zeros(3)
                for t in 1:SEQ_LEN
                    x_b .= @view X_batch[b_idx, t, :]
                end
                x_b ./= SEQ_LEN

                for s in 1:N_SLICES
                    d_proj = clamp.(d_avg_batch, -0.5, 0.5)
                    model.slice_proj_ws[s] .-= lr .* (d_proj * x_b')
                    model.slice_proj_bs[s] .-= lr .* d_proj
                end
            end

            slice_loads = slice_selection_counts ./ max(sum(slice_selection_counts), 1)
            entropy_grad = zeros(4)
            for k in 1:4
                if slice_loads[k] > 1e-8
                    entropy_grad[k] = -(log(slice_loads[k]) + 1.0)
                end
            end

            ec_sel_d_w = zeros(size(model.router.ec_selection_w))
            ec_sel_d_b = zeros(4)
            for s in 1:N_SLICES
                ec_sel_d_w[s, :] .+= entropy_grad[s] * 0.01
                ec_sel_d_b[s] += entropy_grad[s] * 0.01
            end
            model.router.ec_selection_w .-= 0.001 .* clamp.(ec_sel_d_w, -1.0, 1.0)
            model.router.ec_selection_b .-= 0.001 .* clamp.(ec_sel_d_b, -1.0, 1.0)

            slice_selection_counts .*= 0.95
            n_b += 1
        end

        epoch_loss /= max(n_b, 1)

        if epoch % 5 == 0 || epoch == 1
            model_copy_saves = (
                output_w = copy(model.output_w),
                output_b = copy(model.output_b),
                slice_ws = [copy(s.weights[1]) for s in model.slices],
                slice_bs = [copy(s.biases[1]) for s in model.slices],
            )

            outputs, _ = forward_sequence(model, X_val)
            pred_labels = [argmax(@view outputs[i, :]) for i in 1:size(outputs, 1)]
            acc = sum(pred_labels .== y_val) / length(y_val)

            sel_dist = slice_selection_counts ./ max(sum(slice_selection_counts), 1)
            balance = round(1.0 - (maximum(sel_dist) - minimum(sel_dist)), digits=3)

            println("  Epoch $(lpad(epoch, 3)): loss=$(round(epoch_loss, digits=4)), " *
                    "val=$(round(acc*100, digits=1))%, " *
                    "soft_sel=$([round(d*100, digits=1) for d in sel_dist]), " *
                    "balance=$balance")

            if acc > best_val + 0.001
                best_val = acc
                patience = 0
            else
                patience += 5
            end

            if patience >= 25
                println("  Early stopping at epoch $epoch")
                break
            end
        end
    end

    println("\n" * "=" ^ 70)
    println("  ROUTING ANALYSIS (FINAL — using standard router)")
    println("=" ^ 70)

    slice_counts = zeros(4)
    phase_counts = zeros(4)
    for i in 1:min(300, size(X_val, 1))
        _, traces_i = forward_sequence(model, @view X_val[i:i, :, :])
        for trace_list in traces_i
            for t_trace in trace_list
                for s in t_trace.slice_history
                    slice_counts[s] += 1
                end
                for p in t_trace.phase_history
                    phase_counts[Int(p)] += 1
                end
            end
        end
    end

    labels = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]
    println("\n  Slice Usage (eval):")
    for (count, label) in zip(slice_counts, labels)
        pct = count / max(sum(slice_counts), 1) * 100
        bar = "█" * ("░" ^ max(0, round(Int, pct / 3)))
        println("    $(rpad(label, 10)) $(lpad(round(pct, digits=1), 5))% $bar")
    end

    println("\n  Phase Usage (eval):")
    for (count, label) in zip(phase_counts, labels)
        pct = count / max(sum(phase_counts), 1) * 100
        bar = "█" * ("░" ^ max(0, round(Int, pct / 3)))
        println("    $(rpad(label, 10)) $(lpad(round(pct, digits=1), 5))% $bar")
    end

    println("\n  EC Selection Weights (after training):")
    for s in 1:4
        w_norm = norm(@view model.router.ec_selection_w[s, :])
        println("    Slice $s selection norm: $(round(w_norm, digits=4))")
    end

    println("\n  Unique slices: $(Int(sum(slice_counts .> 0)))/4")
    println("  Unique phases: $(Int(sum(phase_counts .> 0)))/4")
    println("  Best val accuracy: $(round(best_val*100, digits=1))%")

    println("\n" * "=" ^ 70)
    println("  Soft Expert Choice experiment complete!")
    println("=" ^ 70)
end

main()
