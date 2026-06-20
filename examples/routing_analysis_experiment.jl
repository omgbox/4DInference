using Random
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference

const SL_OUT = 8
const N_SL = 4

function generate_temporal_task(n_samples::Int; seq_len::Int=8, seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister(42)
    X = Array{Float64}(undef, n_samples, seq_len, 3)
    y = Vector{Int}(undef, n_samples)
    for i in 1:n_samples
        task_type = rand(rng)
        if task_type < 0.25
            direction = rand(rng) > 0.5 ? 1.0 : -1.0
            start_val = rand(rng) * 0.4 + 0.3
            for t in 1:seq_len
                trend = direction * (t - 1) / seq_len * 0.5
                X[i, t, :] .= clamp.([
                    start_val + trend + randn(rng) * 0.03,
                    0.5 + randn(rng) * 0.05,
                    0.5 + randn(rng) * 0.05
                ], 0.0, 1.0)
            end
            y[i] = direction > 0 ? 1 : 2
        elseif task_type < 0.5
            spike_pos = rand(rng, 1:seq_len)
            for t in 1:seq_len
                spike = t == spike_pos ? 0.8 : 0.2
                X[i, t, :] .= clamp.([
                    0.5 + randn(rng) * 0.05,
                    spike + randn(rng) * 0.03,
                    0.5 + randn(rng) * 0.05
                ], 0.0, 1.0)
            end
            y[i] = 3
        elseif task_type < 0.75
            base = rand(rng) * 0.3 + 0.35
            for t in 1:seq_len
                noise1 = sin(2π * t / seq_len) * 0.2
                noise2 = cos(2π * t / seq_len) * 0.2
                X[i, t, :] .= clamp.([
                    base + noise1 + randn(rng) * 0.02,
                    base + noise2 + randn(rng) * 0.02,
                    base + (noise1 + noise2) / 2 + randn(rng) * 0.02
                ], 0.0, 1.0)
            end
            y[i] = 4
        else
            high_phase = rand(rng, 2:3)
            for t in 1:seq_len
                phase_ratio = t / seq_len
                base_val = phase_ratio < high_phase / seq_len ? 0.3 : 0.7
                X[i, t, :] .= clamp.([
                    base_val + randn(rng) * 0.05,
                    base_val + randn(rng) * 0.05,
                    base_val + randn(rng) * 0.05
                ], 0.0, 1.0)
            end
            y[i] = 5 + (high_phase > 2 ? 1 : 0)
        end
    end
    return X, y
end

function softmax_stable(x::AbstractVector{Float64})
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    s = sum(exp_x)
    (s == 0.0 || !isfinite(s)) && return ones(length(x)) ./ length(x)
    exp_x ./ s
end

function forward_sequence_with_trace(model::FourDModel, X_seq::AbstractArray{Float64,3})
    batch_size = size(X_seq, 1)
    seq_len = size(X_seq, 2)
    outputs = Matrix{Float64}(undef, batch_size, size(model.output_w, 1))
    # router_log[b] = [chosen_slice_1_timestep_1, chosen_slice_2_timestep_1, ...]
    router_log = Vector{Int}[]

    for b in 1:batch_size
        accumulated_state = zeros(model.slice_output_dim)
        b_slices = Int[]

        for t in 1:seq_len
            x_t = @view X_seq[b, t, :]
            mem_state = read_memory(model.memory)
            phase = RETRIEVE

            for step in 1:model.max_steps
                phase_embed = encode_phase(model.phase_manager, phase)
                router_input = vcat(x_t, mem_state, phase_embed)
                route = router_forward(model.router, router_input)
                chosen_slice = route.chosen_slice
                chosen_phase, _ = decode_phase(model.phase_manager, route.phase_logits)

                slice = model.slices[chosen_slice]
                projected = model.slice_proj_ws[chosen_slice] * x_t .+ model.slice_proj_bs[chosen_slice]
                raw_output = slice_forward(slice, projected)
                film = model.film_layers[chosen_slice]
                conditioned = film_forward(film, raw_output, phase_embed)
                accumulated_state .+= conditioned
                push!(b_slices, chosen_slice)

                update_phase!(model.phase_manager, chosen_phase)
                phase = chosen_phase
                surprise_write!(model.memory, conditioned)
                mem_state = read_memory(model.memory)

                if route.confidence > 0.8
                    break
                end
            end
        end

        accumulated_state ./= seq_len
        outputs[b, :] .= model.output_w * accumulated_state .+ model.output_b
        push!(router_log, b_slices)
    end

    return outputs, router_log
end

function compute_slice_logits_from_input(model::FourDModel, x_t::AbstractVector{Float64})
    mem_state = read_memory(model.memory)
    phase_embed = encode_phase(model.phase_manager, RETRIEVE)
    router_input = vcat(x_t, mem_state, phase_embed)
    route = router_forward(model.router, router_input)
    return route
end

function input_statistics(X::AbstractArray{Float64,3})
    batch_size = size(X, 1)
    seq_len = size(X, 2)
    stats = Matrix{Float64}(undef, batch_size, 4)

    for i in 1:batch_size
        ch1 = @view X[i, :, 1]
        ch2 = @view X[i, :, 2]
        ch3 = @view X[i, :, 3]
        stats[i, 1] = std(ch1)
        stats[i, 2] = std(ch2)
        stats[i, 3] = std(ch3)
        stats[i, 4] = maximum(X[i, :, :]) - minimum(X[i, :, :])
    end
    return stats
end

function main()
    SEQ_LEN = 8
    N_TRAIN = 2000
    N_VAL = 500
    EPOCHS = 80
    BATCH_SIZE = 32

    println("=" ^ 70)
    println("  ROUTING ANALYSIS ON TEMPORAL TASK (82.1% baseline)")
    println("=" ^ 70)

    X_train, y_train = generate_temporal_task(N_TRAIN; seed=42)
    X_val, y_val = generate_temporal_task(N_VAL; seed=123)

    println("  6 classes — temporal patterns")
    println("  Training: $(size(X_train)), Val: $(size(X_val))")

    model = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=3, n_classes=6, seed=42)

    rng = MersenneTwister(42)
    best_val = 0.0
    best_model_state = nothing
    patience = 0

    routing_history = Vector{Vector{Float64}}()

    for epoch in 1:EPOCHS
        lr = max(0.0005, 0.003 * (1.0 - epoch / EPOCHS))
        indices = shuffle(rng, 1:N_TRAIN)
        epoch_loss = 0.0
        n_b = 0

        # Track which slices are chosen
        slice_chosen = zeros(N_SL)

        for start_idx in 1:BATCH_SIZE:N_TRAIN
            end_idx = min(start_idx + BATCH_SIZE - 1, N_TRAIN)
            batch_idx = indices[start_idx:end_idx]
            X_batch = X_train[batch_idx, :, :]
            y_batch = y_train[batch_idx]
            bs = size(X_batch, 1)

            for b in 1:bs
                accumulated_state = zeros(SL_OUT)
                fwd_slices = Int[]
                fwd_caches = NamedTuple[]
                fwd_router_probs = Vector{Float64}[]

                for t in 1:SEQ_LEN
                    x_t = @view X_batch[b, t, :]
                    mem_state = read_memory(model.memory)
                    phase = RETRIEVE

                    for step in 1:model.max_steps
                        phase_embed = encode_phase(model.phase_manager, phase)
                        router_input = vcat(x_t, mem_state, phase_embed)
                        route = router_forward(model.router, router_input)
                        chosen_slice = route.chosen_slice
                        chosen_phase, _ = decode_phase(model.phase_manager, route.phase_logits)

                        slice_chosen[chosen_slice] += 1

                        projected = model.slice_proj_ws[chosen_slice] * x_t .+ model.slice_proj_bs[chosen_slice]
                        raw_out, acts, pre_acts = slice_forward_cached(model.slices[chosen_slice], projected)
                        film = model.film_layers[chosen_slice]
                        gamma = film.gamma_w * phase_embed .+ film.gamma_b
                        beta = film.beta_w * phase_embed .+ film.beta_b
                        conditioned = gamma .* raw_out .+ beta

                        accumulated_state .+= conditioned
                        push!(fwd_slices, chosen_slice)
                        push!(fwd_router_probs, copy(route.slice_probs))
                        push!(fwd_caches, (
                            acts=acts, pre_acts=pre_acts, gamma=gamma, beta=beta,
                            raw_out=raw_out, phase_embed=phase_embed, slice=chosen_slice,
                            projected=copy(projected), x_t=copy(x_t), phase=phase
                        ))

                        update_phase!(model.phase_manager, chosen_phase)
                        phase = chosen_phase
                        surprise_write!(model.memory, conditioned)
                        mem_state = read_memory(model.memory)

                        if route.confidence > 0.8
                            break
                        end
                    end
                end

                accumulated_state ./= max(length(fwd_slices), 1)
                logits = model.output_w * accumulated_state .+ model.output_b
                probs = softmax_stable(logits)

                target = y_batch[b]
                loss = -log(max(probs[target], 1e-8))
                epoch_loss += loss

                d_logits = copy(probs)
                d_logits[target] -= 1.0
                d_logits = clamp.(d_logits, -1.0, 1.0)

                d_out_w = d_logits * accumulated_state'
                model.output_w .-= lr .* clamp.(d_out_w, -1.0, 1.0)
                model.output_b .-= lr .* clamp.(d_logits, -1.0, 1.0)

                d_avg = model.output_w' * d_logits
                d_avg = clamp.(d_avg, -2.0, 2.0)
                d_per_step = d_avg ./ max(length(fwd_slices), 1)

                for step_idx in 1:length(fwd_slices)
                    si = fwd_slices[step_idx]
                    cache = fwd_caches[step_idx]
                    slice = model.slices[si]
                    film = model.film_layers[si]

                    d_cond = clamp.(d_per_step, -1.0, 1.0)
                    d_gamma = clamp.(d_cond .* cache.raw_out, -1.0, 1.0)
                    d_beta = clamp.(d_cond, -1.0, 1.0)
                    d_raw = clamp.(d_cond .* cache.gamma, -1.0, 1.0)

                    film.gamma_w .-= lr .* (d_gamma * cache.phase_embed')
                    film.gamma_b .-= lr .* d_gamma
                    film.beta_w .-= lr .* (d_beta * cache.phase_embed')
                    film.beta_b .-= lr .* d_beta

                    n_layers = length(slice.weights)
                    saved_ws = [copy(W) for W in slice.weights]
                    delta = d_raw
                    for li in n_layers:-1:1
                        dW = delta * cache.acts[li]'
                        dW = clamp.(dW, -0.5, 0.5)
                        slice.weights[li] .-= lr .* dW
                        slice.biases[li] .-= lr .* clamp.(delta, -0.5, 0.5)
                        if li > 1
                            delta = (saved_ws[li]' * delta) .* (cache.pre_acts[li-1] .> 0.0)
                            delta = clamp.(delta, -2.0, 2.0)
                        end
                    end
                    d_input = n_layers > 1 ? (saved_ws[1]' * delta) : delta
                    d_input = clamp.(d_input, -2.0, 2.0)
                    model.slice_proj_ws[si] .-= lr .* clamp.(d_input * cache.x_t', -0.5, 0.5)
                    model.slice_proj_bs[si] .-= lr .* clamp.(d_input, -0.5, 0.5)

                    # Entropy bonus for router
                    sp = fwd_router_probs[step_idx]
                    entropy_grad = zeros(N_SL)
                    for k in 1:N_SL
                        if sp[k] > 1e-8
                            entropy_grad[k] = -(log(sp[k]) + 1.0)
                        end
                    end

                    # Recompute router input
                    x_t_live = cache.x_t
                    mem_state_live = read_memory(model.memory)
                    phase_embed_r = encode_phase(model.phase_manager, cache.phase)
                    router_input_r = vcat(x_t_live, mem_state_live, phase_embed_r)

                    h1_r = model.router.W1 * router_input_r .+ model.router.b1
                    h2_relu_r = max.(0.0, model.router.W2 * max.(0.0, h1_r) .+ model.router.b2)
                    router_grad = 0.1 .* entropy_grad
                    router_grad = clamp.(router_grad, -5.0, 5.0)

                    d_h2 = model.router.slice_head_w' * router_grad
                    d_h2 = clamp.(d_h2, -1.0, 1.0)
                    d_h2 .*= (h2_relu_r .> 0.0)
                    h1_relu_r = max.(0.0, h1_r)
                    model.router.W2 .-= lr .* 0.5 .* clamp.(d_h2 * h1_relu_r', -1.0, 1.0)
                    model.router.b2 .-= lr .* 0.5 .* clamp.(d_h2, -1.0, 1.0)
                    model.router.slice_head_w .-= lr .* 0.5 .* clamp.(router_grad * h2_relu_r', -0.1, 0.1)
                    model.router.slice_head_b .-= lr .* 0.5 .* clamp.(router_grad, -0.1, 0.1)
                end
            end
            n_b += 1
        end

        epoch_loss /= max(n_b, 1)

        if epoch % 5 == 0 || epoch == 1
            outputs, router_log = forward_sequence_with_trace(model, X_val)

            pred_labels = [argmax(@view outputs[i, :]) for i in 1:size(outputs, 1)]
            acc = sum(pred_labels .== y_val) / length(y_val)

            # Analyze routing
            slice_dist = zeros(N_SL)
            n_router_calls = 0
            for b_slices in router_log
                for si in b_slices
                    slice_dist[si] += 1
                    n_router_calls += 1
                end
            end
            sel_dist = n_router_calls > 0 ? slice_dist ./ n_router_calls : zeros(N_SL)

            # Entropy of routing distribution
            router_entropy = 0.0
            for p in sel_dist
                p > 1e-8 && (router_entropy -= p * log(p))
            end
            router_entropy /= log(N_SL)

            push!(routing_history, copy(sel_dist))

            println("  Epoch $(lpad(epoch, 3)): loss=$(round(epoch_loss, digits=4)), " *
                    "val=$(round(acc*100, digits=1))%, " *
                    "sel=$([round(d*100, digits=1) for d in sel_dist]), " *
                    "H=$(round(router_entropy, digits=3))")

            if acc > best_val + 0.001
                best_val = acc
                patience = 0
            else
                patience += 5
            end
            if patience >= 40
                println("  Early stopping at epoch $epoch")
                break
            end
        end
    end

    println("\n" * "=" ^ 70)
    println("  ROUTING DIVERSITY ANALYSIS")
    println("=" ^ 70)
    println("  Temporal task best accuracy: $(round(best_val*100, digits=1))%")
    println("  Random baseline: 16.7%")

    labels_v = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]
    total_hist = sum(routing_history)
    avg_dist = total_hist ./ max(sum(total_hist), 1.0)
    println("\n  Average routing distribution over all epochs:")
    for (d, label) in zip(avg_dist, labels_v)
        println("    $(rpad(label, 10)) $(lpad(round(d*100, digits=1), 5))%")
    end

    # Analyze which epoch routing first diversified
    first_diverse = 0
    for (ei, dist) in enumerate(routing_history)
        if sum(dist .> 0.1) >= 2
            first_diverse = ei * 5
            break
        end
    end
    println("\n  First epoch with ≥2 slices >10%: epoch $first_diverse")

    # Correlation between input stats and slice choice
    println("\n  --- Input analysis: what drives slice choice? ---")
    outputs_test, router_log_test = forward_sequence_with_trace(model, X_val)
    input_stats = input_statistics(X_val)

    # For each sample, get the most-used slice
    most_used_slice = zeros(Int, size(X_val, 1))
    for (bi, b_slices) in enumerate(router_log_test)
        slice_votes = zeros(N_SL)
        for si in b_slices
            slice_votes[si] += 1
        end
        most_used_slice[bi] = argmax(slice_votes)
    end

    # Stats per slice
    for si in 1:N_SL
        idx = findall(most_used_slice .== si)
        if length(idx) > 5
            mean_stds = mean(input_stats[idx, :], dims=1)
            println("  Slice $si ($(length(idx)) samples): " *
                    "ch1_std=$(round(mean_stds[1], digits=3)), " *
                    "ch2_std=$(round(mean_stds[2], digits=3)), " *
                    "ch3_std=$(round(mean_stds[3], digits=3)), " *
                    "range=$(round(mean_stds[4], digits=3))")
        end
    end

    # Per-class routing analysis
    println("\n  --- Per-class routing ---")
    for c in 1:6
        idx = findall(y_val .== c)
        if length(idx) > 0
            slice_votes_class = zeros(N_SL)
            for bi in idx
                _, routes = forward_sequence_with_trace(model, @view X_val[bi:bi, :, :])
                for b_slices in routes
                    for si in b_slices
                        slice_votes_class[si] += 1
                    end
                end
            end
            total_votes = sum(slice_votes_class)
            if total_votes > 0
                pcts = [round(v/total_votes*100, digits=1) for v in slice_votes_class]
                println("  Class $c: $pcts")
            end
        end
    end

    println("\n" * "=" ^ 70)
    println("  Analysis complete!")
    println("=" ^ 70)
end

main()
