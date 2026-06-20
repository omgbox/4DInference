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

function softmax_stable(x::AbstractVector{Float64})
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    s = sum(exp_x)
    (s == 0.0 || !isfinite(s)) && return ones(length(x)) ./ length(x)
    exp_x ./ s
end

function softmax_stable!(x::AbstractVector{Float64})
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    s = sum(exp_x)
    if s == 0.0 || !isfinite(s)
        x .= 1.0 / length(x)
    else
        x ./= s
    end
    return x
end

function main()
    SEQ_LEN = 8
    N_TRAIN = 2000
    N_VAL = 500
    EPOCHS = 120
    BATCH_SIZE = 32
    TOTAL_HIDDEN = 32

    println("=" ^ 70)
    println("  OUTPUT-DIM PARTITIONED ROUTING")
    println("=" ^ 70)
    println("  KEY INSIGHT: Each slice writes to a DIFFERENT subset of")
    println("  the output representation. Output layer MUST use all slices.")
    println()
    println("  Slice 1 (RETRIEVE):  writes dims 1:8   (ch1-focused proj)")
    println("  Slice 2 (REASON):    writes dims 9:16  (ch2-focused proj)")
    println("  Slice 3 (PLAN):      writes dims 17:24 (ch3-focused proj)")
    println("  Slice 4 (COMPRESS):  writes dims 25:32 (ch1+2 proj)")
    println()
    println("  Representation is 32-dim, partitioned into 4 x 8-dim slots.")
    println("  Router selects which slot to WRITE to each timestep.")
    println()

    model = create_model!(3; hidden_dim=48, memory_neurons=60, max_steps=3, n_classes=12, seed=42)
    SLICE_OUT = 8
    N_SLICES = 4
    REPR_DIM = SLICE_OUT * N_SLICES
    feature_masks = [
        Bool[1, 0, 0],
        Bool[0, 1, 0],
        Bool[0, 0, 1],
        Bool[1, 1, 0],
    ]

    slice_proj_ws_raw = Vector{Matrix{Float64}}()
    slice_proj_bs_raw = Vector{Vector{Float64}}()
    rng_proj = MersenneTwister(99)
    slice_in_dims = [sum(m) for m in feature_masks]
    for i in 1:N_SLICES
        push!(slice_proj_ws_raw, randn(rng_proj, SLICE_OUT, slice_in_dims[i]) .* sqrt(2.0 / slice_in_dims[i]))
        push!(slice_proj_bs_raw, zeros(SLICE_OUT))
    end

    rng_out = MersenneTwister(77)
    output_w = randn(rng_out, 12, REPR_DIM) .* 0.01
    output_b = zeros(12)

    println("  Feature masks:")
    for i in 1:4
        chs = join([j for j in 1:3 if feature_masks[i][j]], ",")
        println("    Slice $i: channels [$chs] → dims $((i-1)*SLICE_OUT+1):$(i*SLICE_OUT)")
    end

    X_train, y_train = generate_multimodal_task(N_TRAIN; seed=42)
    X_val, y_val = generate_multimodal_task(N_VAL; seed=123)
    println("  Training: $(size(X_train)), Val: $(size(X_val))")

    rng = MersenneTwister(42)
    best_val = 0.0
    patience = 0

    println("\n  --- Training with output-partitioned routing ---")

    for epoch in 1:EPOCHS
        lr = max(0.0005, 0.003 * (1.0 - epoch / EPOCHS))
        lr_router = lr * 0.3
        entropy_w = min(0.5, 0.1 + 0.005 * epoch)
        indices = shuffle(rng, 1:N_TRAIN)
        epoch_loss = 0.0
        slice_hits = zeros(N_SLICES)
        n_b = 0

        for start_idx in 1:BATCH_SIZE:N_TRAIN
            end_idx = min(start_idx + BATCH_SIZE - 1, N_TRAIN)
            batch_idx = indices[start_idx:end_idx]
            X_batch = X_train[batch_idx, :, :]
            y_batch = y_train[batch_idx]
            bs = size(X_batch, 1)

            for b in 1:bs
                repr_accumulated = zeros(REPR_DIM)
                fwd_slices = Int[]
                fwd_router_logits = Vector{Float64}[]
                fwd_mem_states = Vector{Float64}[]
                fwd_phase_embeds = Vector{Float64}[]
                fwd_x_ts = Vector{Float64}[]

                for t in 1:SEQ_LEN
                    x_t = @view X_batch[b, t, :]
                    mem_state = read_memory(model.memory)
                    phase_embed = encode_phase(model.phase_manager, RETRIEVE)
                    router_input = vcat(x_t, mem_state, phase_embed)

                    route = router_forward(model.router, router_input)
                    chosen_slice = route.chosen_slice
                    chosen_phase, _ = decode_phase(model.phase_manager, route.phase_logits)

                    slice_hits[chosen_slice] += 1

                    mask = feature_masks[chosen_slice]
                    x_masked = x_t[mask]
                    projected = slice_proj_ws_raw[chosen_slice] * x_masked .+ slice_proj_bs_raw[chosen_slice]

                    slice = model.slices[chosen_slice]
                    raw_out = slice_forward(slice, projected)
                    film = model.film_layers[chosen_slice]
                    conditioned = film_forward(film, raw_out, phase_embed)

                    offset = (chosen_slice - 1) * SLICE_OUT
                    repr_accumulated[offset+1:offset+SLICE_OUT] .+= conditioned

                    push!(fwd_slices, chosen_slice)
                    push!(fwd_router_logits, copy(route.slice_probs))
                    push!(fwd_mem_states, copy(mem_state))
                    push!(fwd_phase_embeds, copy(phase_embed))
                    push!(fwd_x_ts, copy(x_t))

                    update_phase!(model.phase_manager, chosen_phase)
                    surprise_write!(model.memory, conditioned)
                    mem_state = read_memory(model.memory)
                end

                repr_accumulated ./= max(length(fwd_slices), 1)
                logits = output_w * repr_accumulated .+ output_b
                probs = softmax_stable(logits)

                target = y_batch[b]
                loss = -log(max(probs[target], 1e-8))

                entropy = 0.0
                for sl in fwd_router_logits
                    for p in sl
                        p > 1e-8 && (entropy -= p * log(p))
                    end
                end
                entropy /= max(length(fwd_router_logits), 1)

                total_loss = loss - entropy_w * entropy
                epoch_loss += loss

                d_logits_out = copy(probs)
                d_logits_out[target] -= 1.0
                d_logits_out = clamp.(d_logits_out, -2.0, 2.0)

                d_out_w = d_logits_out * repr_accumulated'
                d_out_w = clamp.(d_out_w, -1.0, 1.0)
                output_w .-= lr .* d_out_w
                output_b .-= lr .* clamp.(d_logits_out, -1.0, 1.0)

                d_repr = output_w' * d_logits_out
                d_repr = clamp.(d_repr, -2.0, 2.0)

                for step_idx in 1:length(fwd_slices)
                    chosen_s = fwd_slices[step_idx]
                    offset = (chosen_s - 1) * SLICE_OUT
                    d_conditioned = d_repr[offset+1:offset+SLICE_OUT] ./ max(length(fwd_slices), 1)
                    d_conditioned = clamp.(d_conditioned, -2.0, 2.0)

                    slice = model.slices[chosen_s]
                    film = model.film_layers[chosen_s]
                    phase_embed = fwd_phase_embeds[step_idx]
                    x_t_step = fwd_x_ts[step_idx]

                    mask = feature_masks[chosen_s]
                    x_masked = x_t_step[mask]
                    projected = slice_proj_ws_raw[chosen_s] * x_masked .+ slice_proj_bs_raw[chosen_s]

                    raw_out, acts, pre_acts = slice_forward_cached(slice, projected)
                    gamma = film.gamma_w * phase_embed .+ film.gamma_b
                    beta = film.beta_w * phase_embed .+ film.beta_b

                    d_gamma = clamp.(d_conditioned .* raw_out, -1.0, 1.0)
                    d_beta = clamp.(d_conditioned, -1.0, 1.0)
                    d_raw = clamp.(d_conditioned .* gamma, -1.0, 1.0)

                    film.gamma_w .-= lr .* (d_gamma * phase_embed')
                    film.gamma_b .-= lr .* d_gamma
                    film.beta_w .-= lr .* (d_beta * phase_embed')
                    film.beta_b .-= lr .* d_beta

                    n_layers = length(slice.weights)
                    saved_ws = [copy(W) for W in slice.weights]
                    delta = d_raw
                    for i in n_layers:-1:1
                        dW = delta * acts[i]'
                        dW = clamp.(dW, -0.5, 0.5)
                        slice.weights[i] .-= lr .* dW
                        slice.biases[i] .-= lr .* clamp.(copy(delta), -0.5, 0.5)
                        if i > 1
                            delta = (saved_ws[i]' * delta) .* (pre_acts[i-1] .> 0.0)
                            delta = clamp.(delta, -2.0, 2.0)
                        end
                    end
                    d_input = n_layers > 1 ? (saved_ws[1]' * delta) : d_raw
                    d_input = clamp.(d_input, -2.0, 2.0)
                    slice_proj_ws_raw[chosen_s] .-= lr .* clamp.(d_input * x_masked', -0.5, 0.5)
                    slice_proj_bs_raw[chosen_s] .-= lr .* clamp.(d_input, -0.5, 0.5)
                end

                n_steps = length(fwd_router_logits)
                for step_idx in 1:n_steps
                    router_logit = fwd_router_logits[step_idx]
                    chosen_s = fwd_slices[step_idx]

                    slice_probs = zeros(N_SLICES)
                    softmax_stable!(slice_probs)
                    slice_probs .= router_logit

                    target_onehot = zeros(N_SLICES)
                    target_onehot[chosen_s] = 1.0

                    d_slice_probs = slice_probs .- target_onehot

                    d_slice_probs .+= entropy_w .* (1.0 .+ log.(max.(slice_probs, 1e-8)))

                    d_slice_probs = clamp.(d_slice_probs, -5.0, 5.0)

                    x_t_r = fwd_x_ts[step_idx]
                    mem_state_r = fwd_mem_states[step_idx]
                    phase_embed_r = fwd_phase_embeds[step_idx]
                    router_input_r = vcat(x_t_r, mem_state_r, phase_embed_r)

                    h1 = model.router.W1 * router_input_r .+ model.router.b1
                    h1_relu = max.(0.0, h1)
                    h2 = model.router.W2 * h1_relu .+ model.router.b2
                    h2_relu = max.(0.0, h2)

                    d_h2 = model.router.slice_head_w' * d_slice_probs
                    d_h2 = clamp.(d_h2, -1.0, 1.0)
                    d_h2 .*= (h2 .> 0.0)
                    model.router.W2 .-= lr_router .* clamp.(d_h2 * h1_relu', -1.0, 1.0)
                    model.router.b2 .-= lr_router .* clamp.(d_h2, -1.0, 1.0)

                    d_h1 = model.router.W2' * d_h2
                    d_h1 = clamp.(d_h1, -1.0, 1.0)
                    d_h1 .*= (h1 .> 0.0)
                    model.router.W1 .-= lr_router .* clamp.(d_h1 * router_input_r', -1.0, 1.0)
                    model.router.b1 .-= lr_router .* clamp.(d_h1, -1.0, 1.0)

                    model.router.slice_head_w .-= lr_router .* clamp.(d_slice_probs * h2_relu', -0.1, 0.1)
                    model.router.slice_head_b .-= lr_router .* clamp.(d_slice_probs, -0.1, 0.1)
                end
            end
            n_b += 1
        end

        epoch_loss /= max(n_b, 1)

        if epoch % 5 == 0 || epoch == 1
            acc, sel_dist = evaluate_partitioned(model, X_val, y_val, feature_masks,
                                                slice_proj_ws_raw, slice_proj_bs_raw,
                                                output_w, output_b)

            println("  Epoch $(lpad(epoch, 3)): loss=$(round(epoch_loss, digits=4)), " *
                    "val=$(round(acc*100, digits=1))%, " *
                    "sel=$([round(d*100, digits=1) for d in sel_dist]), " *
                    "ew=$(round(entropy_w, digits=3))")

            if acc > best_val + 0.001
                best_val = acc
                patience = 0
            else
                patience += 5
            end

            if patience >= 50
                println("  Early stopping at epoch $epoch")
                break
            end
        end
    end

    println("\n" * "=" ^ 70)
    println("  FINAL RESULTS")
    println("=" ^ 70)

    acc, sel_dist = evaluate_partitioned(model, X_val, y_val, feature_masks,
                                         slice_proj_ws_raw, slice_proj_bs_raw,
                                         output_w, output_b)

    println("  Output-partitioned accuracy: $(round(acc*100, digits=1))%")
    println("  Router distribution: $([round(d*100, digits=1) for d in sel_dist])")
    println("  Previous best (no capacity-lock): 55.0%")
    println("  Previous best (temporal only): 82.1%")
    println("  Random baseline: $(round(100/12, digits=1))%")

    slice_counts = zeros(N_SLICES)
    for i in 1:min(500, size(X_val, 1))
        for t in 1:SEQ_LEN
            x_t = @view X_val[i, t, :]
            mem_state = read_memory(model.memory)
            phase_embed = encode_phase(model.phase_manager, RETRIEVE)
            router_input = vcat(x_t, mem_state, phase_embed)
            route = router_forward(model.router, router_input)
            slice_counts[route.chosen_slice] += 1
            mask = feature_masks[route.chosen_slice]
            x_masked = x_t[mask]
            projected = slice_proj_ws_raw[route.chosen_slice] * x_masked .+ slice_proj_bs_raw[route.chosen_slice]
            slice = model.slices[route.chosen_slice]
            raw_out = slice_forward(slice, projected)
            film = model.film_layers[route.chosen_slice]
            conditioned = film_forward(film, raw_out, phase_embed)
            surprise_write!(model.memory, conditioned)
            update_phase!(model.phase_manager, RETRIEVE)
        end
    end

    labels = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]
    println("\n  Final Slice Usage:")
    for (count, label) in zip(slice_counts, labels)
        pct = count / max(sum(slice_counts), 1) * 100
        bar = "█" * ("░" ^ max(0, round(Int, pct / 3)))
        println("    $(rpad(label, 10)) $(lpad(round(pct, digits=1), 5))% $bar")
    end
    println("  Unique slices: $(Int(sum(slice_counts .> 0)))/4")
    println("=" ^ 70)
end

function evaluate_partitioned(model, X, y, feature_masks, slice_proj_ws, slice_proj_bs, output_w, output_b)
    batch_size = size(X, 1)
    seq_len = size(X, 2)
    N_SLICES = 4
    SLICE_OUT = 8
    REPR_DIM = SLICE_OUT * N_SLICES
    outputs_mat = Matrix{Float64}(undef, batch_size, size(output_w, 1))
    slice_counts = zeros(N_SLICES)

    for b in 1:batch_size
        repr_accum = zeros(REPR_DIM)
        for t in 1:seq_len
            x_t = @view X[b, t, :]
            mem_state = read_memory(model.memory)
            phase_embed = encode_phase(model.phase_manager, RETRIEVE)
            router_input = vcat(x_t, mem_state, phase_embed)
            route = router_forward(model.router, router_input)
            chosen_slice = route.chosen_slice
            chosen_phase, _ = decode_phase(model.phase_manager, route.phase_logits)
            slice_counts[chosen_slice] += 1

            mask = feature_masks[chosen_slice]
            x_masked = x_t[mask]
            projected = slice_proj_ws[chosen_slice] * x_masked .+ slice_proj_bs[chosen_slice]
            slice = model.slices[chosen_slice]
            raw_output = slice_forward(slice, projected)
            film = model.film_layers[chosen_slice]
            conditioned = film_forward(film, raw_output, phase_embed)

            offset = (chosen_slice - 1) * SLICE_OUT
            repr_accum[offset+1:offset+SLICE_OUT] .+= conditioned

            update_phase!(model.phase_manager, chosen_phase)
            surprise_write!(model.memory, conditioned)
            mem_state = read_memory(model.memory)
        end
        repr_accum ./= seq_len
        outputs_mat[b, :] .= output_w * repr_accum .+ output_b
    end

    pred_labels = [argmax(@view outputs_mat[i, :]) for i in 1:batch_size]
    acc = sum(pred_labels .== y) / length(y)
    total = sum(slice_counts)
    sel_dist = total > 0 ? slice_counts ./ total : ones(N_SLICES) ./ N_SLICES
    return acc, sel_dist
end

main()
