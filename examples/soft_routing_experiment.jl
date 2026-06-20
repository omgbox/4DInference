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

function nan_guard!(w, label="")
    if any(!isfinite, w)
        println("  WARNING: NaN/Inf in $label, clamping")
        replace!(w, NaN=>0.0, Inf=>1.0, -Inf=>-1.0)
        return true
    end
    return false
end

const SL_OUT = 8
const N_SL = 4

function main()
    SEQ_LEN = 8
    N_TRAIN = 2000
    N_VAL = 500
    EPOCHS = 150
    BATCH_SIZE = 32

    println("=" ^ 70)
    println("  SOFT ROUTING V2 — Weighted combination, no LFB, NaN guards")
    println("=" ^ 70)

    X_train, y_train = generate_multimodal_task(N_TRAIN; seed=42)
    X_val, y_val = generate_multimodal_task(N_VAL; seed=123)
    println("  4 modalities x 3 subclasses = 12 classes")
    println("  Training: $(size(X_train)), Val: $(size(X_val))")

    model = create_model!(3; hidden_dim=48, memory_neurons=60, max_steps=3, n_classes=12, seed=42)

    rng = MersenneTwister(42)
    best_val = 0.0
    patience = 0

    for epoch in 1:EPOCHS
        lr = max(0.0003, 0.002 * (1.0 - epoch / EPOCHS))
        lr_r = lr * 0.5  # Router learns slower
        indices = shuffle(rng, 1:N_TRAIN)
        epoch_loss = 0.0
        n_b = 0
        routing_weights = Float64[]

        for start_idx in 1:BATCH_SIZE:N_TRAIN
            end_idx = min(start_idx + BATCH_SIZE - 1, N_TRAIN)
            batch_idx = indices[start_idx:end_idx]
            X_batch = X_train[batch_idx, :, :]
            y_batch = y_train[batch_idx]
            bs = size(X_batch, 1)

            for b in 1:bs
                acc_state = zeros(SL_OUT)
                n_t = 0

                # Store per-timestep data for backprop
                t_router_probs = Vector{Float64}[]
                t_router_inputs = Vector{Float64}[]
                t_slice_outputs = Vector{Vector{Float64}}[]
                t_fwd_caches = Vector{NamedTuple}[]
                t_x_ts = Vector{Float64}[]
                t_phase_embeds = Vector{Float64}[]

                for t in 1:SEQ_LEN
                    x_t = @view X_batch[b, t, :]
                    mem_state = read_memory(model.memory)
                    phase_embed = encode_phase(model.phase_manager, RETRIEVE)
                    router_input = vcat(x_t, mem_state, phase_embed)

                    # Router forward
                    h1 = model.router.W1 * router_input .+ model.router.b1
                    h1_relu = max.(0.0, h1)
                    h2 = model.router.W2 * h1_relu .+ model.router.b2
                    h2_relu = max.(0.0, h2)
                    slice_logits = model.router.slice_head_w * h2_relu .+ model.router.slice_head_b
                    slice_probs = softmax_stable(slice_logits)

                    # All slices process input
                    slice_outputs = Vector{Float64}[]
                    fwd_caches = NamedTuple[]
                    combined = zeros(SL_OUT)

                    for si in 1:N_SL
                        mask = trues(3)
                        x_masked = x_t[mask]
                        projected = model.slice_proj_ws[si] * x_masked .+ model.slice_proj_bs[si]
                        raw_out, acts, pre_acts = slice_forward_cached(model.slices[si], projected)
                        film = model.film_layers[si]
                        gamma = film.gamma_w * phase_embed .+ film.gamma_b
                        beta = film.beta_w * phase_embed .+ film.beta_b
                        conditioned = gamma .* raw_out .+ beta

                        push!(slice_outputs, copy(conditioned))
                        push!(fwd_caches, (
                            acts=acts, pre_acts=pre_acts, gamma=gamma, beta=beta,
                            raw_out=raw_out, projected=projected, si=si,
                            x_masked=copy(x_masked), phase_embed=copy(phase_embed)
                        ))
                        combined .+= slice_probs[si] .* conditioned
                    end

                    acc_state .+= combined
                    n_t += 1
                    push!(t_router_probs, copy(slice_probs))
                    push!(t_router_inputs, copy(router_input))
                    push!(t_slice_outputs, slice_outputs)
                    push!(t_fwd_caches, fwd_caches)
                    push!(t_x_ts, copy(x_t))
                    push!(t_phase_embeds, copy(phase_embed))
                    push!(routing_weights, argmax(slice_probs))
                end

                acc_state ./= n_t
                logits = model.output_w * acc_state .+ model.output_b
                probs = softmax_stable(logits)
                target = y_batch[b]
                loss = -log(max(probs[target], 1e-8))

                any(!isfinite, probs) && continue
                epoch_loss += loss

                # Output layer gradient
                d_logits = copy(probs)
                d_logits[target] -= 1.0
                d_logits = clamp.(d_logits, -2.0, 2.0)

                d_out_w = d_logits * acc_state'
                d_out_w = clamp.(d_out_w, -1.0, 1.0)
                model.output_w .-= lr .* d_out_w
                model.output_b .-= lr .* clamp.(d_logits, -1.0, 1.0)

                # Gradient to accumulated state
                d_acc = model.output_w' * d_logits
                d_acc = clamp.(d_acc, -2.0, 2.0)
                d_per_t = d_acc ./ n_t

                # Accumulate gradients across timesteps, then apply
                avg_slice_grads_w = [zeros(size(model.slice_proj_ws[si])) for si in 1:N_SL]
                avg_slice_grads_b = [zeros(length(model.slice_proj_bs[si])) for si in 1:N_SL]
                avg_slice_weight_grads = [[zeros(size(model.slices[si].weights[li])) for li in 1:length(model.slices[si].weights)] for si in 1:N_SL]
                avg_slice_bias_grads = [[zeros(length(model.slices[si].biases[li])) for li in 1:length(model.slices[si].biases)] for si in 1:N_SL]
                avg_film_gamma_w = [zeros(size(model.film_layers[si].gamma_w)) for si in 1:N_SL]
                avg_film_gamma_b = [zeros(length(model.film_layers[si].gamma_b)) for si in 1:N_SL]
                avg_film_beta_w = [zeros(size(model.film_layers[si].beta_w)) for si in 1:N_SL]
                avg_film_beta_b = [zeros(length(model.film_layers[si].beta_b)) for si in 1:N_SL]
                avg_router_W1 = zeros(size(model.router.W1))
                avg_router_b1 = zeros(length(model.router.b1))
                avg_router_W2 = zeros(size(model.router.W2))
                avg_router_b2 = zeros(length(model.router.b2))
                avg_router_sw = zeros(size(model.router.slice_head_w))
                avg_router_sb = zeros(length(model.router.slice_head_b))

                for t_idx in 1:n_t
                    slice_probs = t_router_probs[t_idx]
                    router_input = t_router_inputs[t_idx]
                    fwd_caches = t_fwd_caches[t_idx]
                    x_t = t_x_ts[t_idx]
                    phase_embed = t_phase_embeds[t_idx]

                    # Router forward for gradient computation
                    h1 = model.router.W1 * router_input .+ model.router.b1
                    h1_relu = max.(0.0, h1)
                    h2 = model.router.W2 * h1_relu .+ model.router.b2
                    h2_relu = max.(0.0, h2)

                    d_combined = d_per_t

                    # Backprop through each slice
                    for si in 1:N_SL
                        cache = fwd_caches[si]
                        w = slice_probs[si]

                        d_conditioned = d_combined .* w
                        d_gamma = clamp.(d_conditioned .* cache.raw_out, -1.0, 1.0)
                        d_beta = clamp.(d_conditioned, -1.0, 1.0)
                        d_raw = clamp.(d_conditioned .* cache.gamma, -1.0, 1.0)

                        avg_film_gamma_w[si] .+= d_gamma * cache.phase_embed'
                        avg_film_gamma_b[si] .+= d_gamma
                        avg_film_beta_w[si] .+= d_beta * cache.phase_embed'
                        avg_film_beta_b[si] .+= d_beta

                        # Backprop through slice MLP
                        n_layers = length(model.slices[si].weights)
                        saved_ws = [copy(W) for W in model.slices[si].weights]
                        delta = d_raw

                        for li in n_layers:-1:1
                            avg_slice_weight_grads[si][li] .+= delta * cache.acts[li]'
                            avg_slice_bias_grads[si][li] .+= delta
                            if li > 1
                                delta = (saved_ws[li]' * delta) .* (cache.pre_acts[li-1] .> 0.0)
                                delta = clamp.(delta, -2.0, 2.0)
                            end
                        end

                        d_input = n_layers > 1 ? (saved_ws[1]' * delta) : delta
                        d_input = clamp.(d_input, -2.0, 2.0)
                        avg_slice_grads_w[si] .+= d_input * cache.x_masked'
                        avg_slice_grads_b[si] .+= d_input
                    end

                    # Router gradient
                    router_d_probs = zeros(N_SL)
                    for si in 1:N_SL
                        outs = t_slice_outputs[t_idx][si]
                        router_d_probs[si] = dot(d_combined, outs)
                    end
                    router_d_probs = clamp.(router_d_probs, -5.0, 5.0)

                    # Entropy bonus
                    entropy_grad = zeros(N_SL)
                    for k in 1:N_SL
                        if slice_probs[k] > 1e-8
                            entropy_grad[k] = -(log(slice_probs[k]) + 1.0)
                        end
                    end
                    router_gradient = router_d_probs .+ 0.3 .* entropy_grad
                    router_gradient = clamp.(router_gradient, -5.0, 5.0)

                    # Softmax gradient: dL/d(logits)
                    softmax_grad = slice_probs .* (router_gradient .- dot(slice_probs, router_gradient))
                    softmax_grad = clamp.(softmax_grad, -5.0, 5.0)

                    # Backprop through router MLP
                    d_h2 = model.router.slice_head_w' * softmax_grad
                    d_h2 = clamp.(d_h2, -1.0, 1.0)
                    d_h2 .*= (h2 .> 0.0)

                    avg_router_W2 .+= d_h2 * h1_relu'
                    avg_router_b2 .+= d_h2

                    d_h1 = model.router.W2' * d_h2
                    d_h1 = clamp.(d_h1, -1.0, 1.0)
                    d_h1 .*= (h1 .> 0.0)

                    avg_router_W1 .+= d_h1 * router_input'
                    avg_router_b1 .+= d_h1

                    avg_router_sw .+= softmax_grad * h2_relu'
                    avg_router_sb .+= softmax_grad
                end

                # Apply accumulated gradients
                scale = 1.0 / n_t
                for si in 1:N_SL
                    model.slice_proj_ws[si] .-= lr .* clamp.(avg_slice_grads_w[si] .* scale, -1.0, 1.0)
                    model.slice_proj_bs[si] .-= lr .* clamp.(avg_slice_grads_b[si] .* scale, -1.0, 1.0)

                    model.film_layers[si].gamma_w .-= lr .* clamp.(avg_film_gamma_w[si] .* scale, -1.0, 1.0)
                    model.film_layers[si].gamma_b .-= lr .* clamp.(avg_film_gamma_b[si] .* scale, -1.0, 1.0)
                    model.film_layers[si].beta_w .-= lr .* clamp.(avg_film_beta_w[si] .* scale, -1.0, 1.0)
                    model.film_layers[si].beta_b .-= lr .* clamp.(avg_film_beta_b[si] .* scale, -1.0, 1.0)

                    n_layers = length(model.slices[si].weights)
                    for li in 1:n_layers
                        model.slices[si].weights[li] .-= lr .* clamp.(avg_slice_weight_grads[si][li] .* scale, -1.0, 1.0)
                        model.slices[si].biases[li] .-= lr .* clamp.(avg_slice_bias_grads[si][li] .* scale, -1.0, 1.0)
                    end
                end

                model.router.W1 .-= lr_r .* clamp.(avg_router_W1 .* scale, -1.0, 1.0)
                model.router.b1 .-= lr_r .* clamp.(avg_router_b1 .* scale, -1.0, 1.0)
                model.router.W2 .-= lr_r .* clamp.(avg_router_W2 .* scale, -1.0, 1.0)
                model.router.b2 .-= lr_r .* clamp.(avg_router_b2 .* scale, -1.0, 1.0)
                model.router.slice_head_w .-= lr_r .* clamp.(avg_router_sw .* scale, -1.0, 1.0)
                model.router.slice_head_b .-= lr_r .* clamp.(avg_router_sb .* scale, -1.0, 1.0)
            end
            n_b += 1
        end

        epoch_loss /= max(n_b, 1)

        if !isfinite(epoch_loss)
            println("  Epoch $epoch: NaN loss, stopping")
            break
        end

        if epoch % 5 == 0 || epoch == 1
            acc, mean_weights = evaluate_soft(model, X_val, y_val)
            println("  Epoch $(lpad(epoch, 3)): loss=$(round(epoch_loss, digits=4)), " *
                    "val=$(round(acc*100, digits=1))%, " *
                    "wts=$([round(w*100, digits=1) for w in mean_weights])")

            if acc > best_val + 0.001
                best_val = acc
                patience = 0
            else
                patience += 5
            end
            if patience >= 60
                println("  Early stopping at epoch $epoch")
                break
            end
        end
    end

    println("\n" * "=" ^ 70)
    println("  FINAL: Soft Routing Analysis")
    println("=" ^ 70)

    acc, mean_weights = evaluate_soft(model, X_val, y_val)
    println("  Accuracy: $(round(acc*100, digits=1))%")
    println("  Previous best (hard routing): 55.0%")
    println("  Random baseline: $(round(100/12, digits=1))%")

    labels_v = ["RETRIEVE", "REASON", "PLAN", "COMPRESS"]
    println("\n  Mean router weights:")
    for (w, label) in zip(mean_weights, labels_v)
        pct = w * 100
        bar = "█" * ("░" ^ max(0, round(Int, pct / 3)))
        println("    $(rpad(label, 10)) $(lpad(round(pct, digits=1), 5))% $bar")
    end
    println("  Effective slices used: $(sum(mean_weights .> 0.05))/4")
    println("=" ^ 70)
end

function evaluate_soft(model, X, y)
    batch_size = size(X, 1)
    seq_len = size(X, 2)
    outputs = Matrix{Float64}(undef, batch_size, size(model.output_w, 1))
    total_weights = zeros(N_SL)

    for b in 1:batch_size
        acc_state = zeros(SL_OUT)
        n_t = 0

        for t in 1:seq_len
            x_t = @view X[b, t, :]
            mem_state = read_memory(model.memory)
            phase_embed = encode_phase(model.phase_manager, RETRIEVE)
            router_input = vcat(x_t, mem_state, phase_embed)

            h1 = model.router.W1 * router_input .+ model.router.b1
            h1_relu = max.(0.0, h1)
            h2 = model.router.W2 * h1_relu .+ model.router.b2
            h2_relu = max.(0.0, h2)
            slice_logits = model.router.slice_head_w * h2_relu .+ model.router.slice_head_b
            slice_probs = softmax_stable(slice_logits)
            total_weights .+= slice_probs

            combined = zeros(SL_OUT)
            for si in 1:N_SL
                mask = trues(3)
                x_masked = x_t[mask]
                projected = model.slice_proj_ws[si] * x_masked .+ model.slice_proj_bs[si]
                raw_out = slice_forward(model.slices[si], projected)
                film = model.film_layers[si]
                conditioned = film_forward(film, raw_out, phase_embed)
                combined .+= slice_probs[si] .* conditioned
            end

            acc_state .+= combined
            n_t += 1
        end

        acc_state ./= n_t
        outputs[b, :] .= model.output_w * acc_state .+ model.output_b
    end

    total_weights ./= (batch_size * seq_len)
    pred_labels = [argmax(@view outputs[i, :]) for i in 1:batch_size]
    acc = sum(pred_labels .== y) / length(y)
    return acc, total_weights
end

main()
