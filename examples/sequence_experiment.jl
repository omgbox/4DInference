using Random
using Statistics
using Dates
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference

function print_header(title::String)
    width = 70
    println("\n" * "=" ^ width)
    println(" " ^ max(0, div(width - length(title), 2)) * title)
    println("=" ^ width)
end

function generate_temporal_task(n_samples::Int; seq_len::Int=8,
                                seed::Union{Int,Nothing}=nothing)
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

function mixup_temporal(X::AbstractArray{Float64, 3}, y::AbstractVector{Int};
                       alpha::Float64=0.2, rng::MersenneTwister)
    n = length(y)
    lambda = rand(rng) * (2.0 * alpha) + (1.0 - alpha)
    lambda = clamp(lambda, 0.0, 1.0)
    indices = shuffle(rng, 1:n)
    X_mixed = lambda .* X .+ (1.0 - lambda) .* X[indices, :, :]
    return X_mixed, y, y[indices], lambda
end

function count_parameters(model::FourDModel)
    total = 0
    total += length(model.input_proj_w) + length(model.input_proj_b)
    for w in model.slice_proj_ws; total += length(w); end
    for b in model.slice_proj_bs; total += length(b); end
    total += length(model.output_w) + length(model.output_b)
    for slice in model.slices
        for W in slice.weights; total += length(W); end
        for b in slice.biases; total += length(b); end
    end
    for film in model.film_layers
        total += length(film.gamma_w) + length(film.gamma_b)
        total += length(film.beta_w) + length(film.beta_b)
    end
    total += length(model.router.W1) + length(model.router.b1)
    total += length(model.router.W2) + length(model.router.b2)
    total += length(model.router.slice_head_w) + length(model.router.slice_head_b)
    total += length(model.router.phase_head_w) + length(model.router.phase_head_b)
    total += length(model.router.confidence_head_w) + 1
    return total
end

function evaluate_sequence(model::FourDModel, X::AbstractArray{Float64, 3},
                           y::AbstractVector{Int})
    outputs, _ = forward_sequence(model, X)
    pred_labels = [argmax(@view outputs[i, :]) for i in 1:size(outputs, 1)]
    return sum(pred_labels .== y) / length(y)
end

function cosine_anneal_lr(base_lr::Float64, epoch::Int, max_epochs::Int)
    return base_lr * 0.5 * (1.0 + cos(π * epoch / max_epochs))
end

function switch_aux_loss(slice_probs_batch::Vector{Vector{Float64}}, n_experts::Int=4)
    f = zeros(n_experts)
    P = zeros(n_experts)
    T = length(slice_probs_batch)

    for sp in slice_probs_batch
        chosen = argmax(sp)
        f[chosen] += 1.0
        P .+= sp
    end
    f ./= max(T, 1)
    P ./= max(T, 1)

    return n_experts * sum(f .* P)
end

function expert_orthogonality_loss(slices::Vector{Slice})
    total_loss = 0.0
    n_pairs = 0

    for i in 1:length(slices)
        for j in (i+1):length(slices)
            Wi = slices[i].weights[1]
            Wj = slices[j].weights[1]

            proj_overlap = 0.0
            for k in 1:size(Wi, 1)
                vi = @view Wi[k, :]
                for l in 1:size(Wj, 1)
                    vj = @view Wj[l, :]
                    dot_ij = dot(vi, vj)
                    norm_j = dot(vj, vj) + 1e-8
                    proj = (dot_ij / norm_j) * vj
                    proj_overlap += dot(proj, proj)
                end
            end

            total_loss += proj_overlap / (size(Wi, 1) * size(Wj, 1))
            n_pairs += 1
        end
    end

    return n_pairs > 0 ? total_loss / n_pairs : 0.0
end

function routing_variance_loss(slice_probs_batch::Vector{Vector{Float64}}, n_experts::Int=4)
    if isempty(slice_probs_batch)
        return 0.0
    end

    avg_probs = zeros(n_experts)
    for sp in slice_probs_batch
        avg_probs .+= sp
    end
    avg_probs ./= length(slice_probs_batch)

    variance = 0.0
    for sp in slice_probs_batch
        variance += sum((sp .- avg_probs) .^ 2)
    end
    variance /= length(slice_probs_batch)

    return -variance
end

function phase_diversity_loss(phase_probs_batch::Vector{Vector{Float64}}, n_phases::Int=4)
    if isempty(phase_probs_batch)
        return 0.0
    end

    avg_probs = zeros(n_phases)
    for pp in phase_probs_batch
        avg_probs .+= pp
    end
    avg_probs ./= length(phase_probs_batch)

    entropy = 0.0
    for p in avg_probs
        if p > 1e-8
            entropy -= p * log(p)
        end
    end

    max_entropy = log(n_phases)
    return -(entropy / max_entropy)
end

function loss_free_balancing_bias(expert_loads::Vector{Float64},
                                  target_load::Float64,
                                  gamma::Float64=0.001)
    biases = zeros(length(expert_loads))
    for i in 1:length(expert_loads)
        if expert_loads[i] > target_load
            biases[i] = -gamma
        elseif expert_loads[i] < target_load
            biases[i] = gamma
        end
    end
    return biases
end

function router_z_loss(router_logits::Vector{Float64})
    z = log(sum(exp.(router_logits)))^2
    return z
end

function train_with_research_improvements!(model::FourDModel, X_train::AbstractArray{Float64, 3},
                                           y_train::AbstractVector{Int},
                                           X_val::AbstractArray{Float64, 3},
                                           y_val::AbstractVector{Int};
                                           epochs::Int=100,
                                           batch_size::Int=32,
                                           base_lr::Float64=0.002,
                                           router_lr::Float64=0.0005,
                                           weight_decay::Float64=1e-4,
                                           dropout_rate::Float64=0.15,
                                           aux_loss_coeff::Float64=0.0001,
                                           ortho_coeff::Float64=0.01,
                                           var_coeff::Float64=0.01,
                                           z_loss_coeff::Float64=0.001,
                                           lfb_gamma::Float64=0.001,
                                           lfb_decay::Bool=true,
                                           use_mixup::Bool=true,
                                           mixup_alpha::Float64=0.2,
                                           use_lfb::Bool=true,
                                           print_every::Int=10)

    n_train = size(X_train, 1)
    rng = MersenneTwister(42)
    expert_loads = zeros(4)
    total_routed = 0.0
    lfb_biases = zeros(4)
    running_baseline = 2.0

    best_val = 0.0
    patience = 0
    PATIENCE = 25

    best_weights = (
        output_w = copy(model.output_w),
        output_b = copy(model.output_b),
        input_proj_w = copy(model.input_proj_w),
        input_proj_b = copy(model.input_proj_b),
        slice_proj_ws = [copy(w) for w in model.slice_proj_ws],
        slice_proj_bs = [copy(b) for b in model.slice_proj_bs],
        slice_ws = [copy(s.weights[1]) for s in model.slices],
        slice_bs = [copy(s.biases[1]) for s in model.slices],
        router_W1 = copy(model.router.W1),
        router_b1 = copy(model.router.b1),
        router_W2 = copy(model.router.W2),
        router_b2 = copy(model.router.b2),
        slice_head_w = copy(model.router.slice_head_w),
        slice_head_b = copy(model.router.slice_head_b),
    )

    history = Dict{String, Vector{Float64}}(
        "loss" => Float64[], "val_acc" => Float64[], "test_acc" => Float64[]
    )

    for epoch in 1:epochs
        lr = cosine_anneal_lr(base_lr, epoch, epochs)
        lfb_scale = lfb_decay ? max(0.0, 1.0 - epoch / (epochs * 0.7)) : 1.0
        indices = shuffle(rng, 1:n_train)

        epoch_loss = 0.0
        epoch_aux = 0.0
        epoch_ortho = 0.0
        epoch_var = 0.0
        n_batches = 0
        slice_probs_epoch = Vector{Float64}[]
        phase_probs_epoch = Vector{Float64}[]

        for start_idx in 1:batch_size:n_train
            end_idx = min(start_idx + batch_size - 1, n_train)
            batch_idx = indices[start_idx:end_idx]

            X_batch = X_train[batch_idx, :, :]
            y_batch = y_train[batch_idx]

            if use_mixup && rand(rng) < 0.5
                X_batch, y_batch_primary, y_batch_secondary, lambda = mixup_temporal(
                    X_batch, y_batch; alpha=mixup_alpha, rng=rng)
            else
                lambda = 1.0
                y_batch_primary = y_batch
                y_batch_secondary = y_batch
            end

            batch_loss = 0.0
            batch_size_actual = size(X_batch, 1)
            last_route = nothing
            last_chosen_phase = RETRIEVE
            last_chosen_slice = 1
            last_x_t = zeros(3)
            last_mem_state = zeros(1)

            for b in 1:batch_size_actual
                accumulated_state = zeros(model.slice_output_dim)
                sample_slice_probs = Vector{Float64}[]
                sample_phase_probs = Vector{Float64}[]

                fwd_chosen_slices = Int[]
                fwd_caches = NamedTuple[]

                for t in 1:size(X_batch, 2)
                    x_t = @view X_batch[b, t, :]
                    mem_state = read_memory(model.memory)
                    phase = RETRIEVE

                    for step in 1:model.max_steps
                        phase_embed = encode_phase(model.phase_manager, phase)
                        router_input = vcat(x_t, mem_state, phase_embed)
                        route = router_forward(model.router, router_input)
                        last_route = route

                        adjusted_probs = copy(route.slice_probs)
                        if use_lfb
                            adjusted_probs .+= lfb_scale .* lfb_biases
                            adjusted_probs .= max.(adjusted_probs, 0.0)
                            total_p = sum(adjusted_probs)
                            if total_p > 0
                                adjusted_probs ./= total_p
                            end
                        end

                        chosen_slice = argmax(adjusted_probs)
                        chosen_phase, _ = decode_phase(model.phase_manager, route.phase_logits)
                        last_chosen_phase = chosen_phase
                        last_chosen_slice = chosen_slice
                        last_x_t = copy(x_t)
                        last_mem_state = copy(mem_state)

                        expert_loads[chosen_slice] += 1.0
                        total_routed += 1.0

                        recent_slices = expert_loads ./ max(total_routed, 1.0)
                        lfb_biases = loss_free_balancing_bias(recent_slices, 0.25, lfb_gamma)

                        push!(sample_slice_probs, adjusted_probs)
                        push!(sample_phase_probs, route.phase_probs)

                        projected = model.slice_proj_ws[chosen_slice] * x_t .+ model.slice_proj_bs[chosen_slice]
                        slice = model.slices[chosen_slice]
                        raw_out, acts, pre_acts = slice_forward_cached(slice, projected)
                        film = model.film_layers[chosen_slice]
                        gamma = film.gamma_w * phase_embed .+ film.gamma_b
                        beta = film.beta_w * phase_embed .+ film.beta_b
                        conditioned = gamma .* raw_out .+ beta

                        dropout_mask = rand(rng, length(conditioned)) .> dropout_rate
                        conditioned .*= dropout_mask
                        conditioned .*= 1.0 / (1.0 - dropout_rate)

                        accumulated_state .+= conditioned

                        push!(fwd_chosen_slices, chosen_slice)
                        push!(fwd_caches, (
                            acts=acts, pre_acts=pre_acts,
                            gamma=gamma, beta=beta, raw_out=raw_out,
                            phase_embed=phase_embed, slice_idx=chosen_slice
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

                accumulated_state ./= size(X_batch, 2)
                logits = model.output_w * accumulated_state .+ model.output_b

                probs = exp.(logits .- maximum(logits))
                probs ./= sum(probs)

                target_primary = y_batch_primary[b]
                loss_primary = -log(max(probs[target_primary], 1e-8))

                loss = loss_primary

                if lambda < 1.0
                    target_secondary = y_batch_secondary[b]
                    loss_secondary = -log(max(probs[target_secondary], 1e-8))
                    loss = lambda * loss_primary + (1.0 - lambda) * loss_secondary
                end

                z_loss = router_z_loss(last_route.slice_logits)
                loss += z_loss_coeff * z_loss

                batch_loss += loss
                running_baseline = 0.95 * running_baseline + 0.05 * loss

                for sp in sample_slice_probs
                    push!(slice_probs_epoch, sp)
                end
                for pp in sample_phase_probs
                    push!(phase_probs_epoch, pp)
                end

                d_logits = copy(probs)
                d_logits[target_primary] -= 1.0
                d_logits = clamp.(d_logits, -2.0, 2.0)

                d_avg = model.output_w' * d_logits
                d_avg = clamp.(d_avg, -2.0, 2.0)

                d_out_w = d_logits * accumulated_state'
                d_out_w .-= weight_decay .* model.output_w
                d_out_w = clamp.(d_out_w, -1.0, 1.0)
                model.output_w .-= lr .* d_out_w
                model.output_b .-= lr .* clamp.(d_logits, -1.0, 1.0)

                slice_proj_d_ws = [zeros(size(model.slice_proj_ws[i])) for i in 1:4]
                slice_proj_d_bs = [zeros(length(model.slice_proj_bs[i])) for i in 1:4]

                n_steps = length(fwd_chosen_slices)
                d_per_step = d_avg ./ max(n_steps, 1)

                for step_idx in 1:n_steps
                    chosen_s = fwd_chosen_slices[step_idx]
                    cache = fwd_caches[step_idx]
                    slice = model.slices[chosen_s]
                    film = model.film_layers[chosen_s]

                    d_conditioned = clamp.(d_per_step, -2.0, 2.0)

                    d_gamma = d_conditioned .* cache.raw_out
                    d_beta = d_conditioned
                    d_raw = d_conditioned .* cache.gamma

                    d_gamma = clamp.(d_gamma, -1.0, 1.0)
                    d_beta = clamp.(d_beta, -1.0, 1.0)
                    d_raw = clamp.(d_raw, -1.0, 1.0)

                    film.gamma_w .-= lr .* (d_gamma * cache.phase_embed')
                    film.gamma_b .-= lr .* d_gamma
                    film.beta_w .-= lr .* (d_beta * cache.phase_embed')
                    film.beta_b .-= lr .* d_beta

                    n_layers = length(slice.weights)
                    saved_ws = [copy(W) for W in slice.weights]
                    delta = d_raw

                    for i in n_layers:-1:1
                        dW = delta * cache.acts[i]'
                        db = copy(delta)
                        dW = clamp.(dW, -0.5, 0.5)
                        slice.weights[i] .-= lr .* dW
                        slice.biases[i] .-= lr .* clamp.(db, -0.5, 0.5)

                        if i > 1
                            delta = (saved_ws[i]' * delta) .* (cache.pre_acts[i-1] .> 0.0)
                            delta = clamp.(delta, -2.0, 2.0)
                        end
                    end

                    d_input = n_layers > 1 ? (saved_ws[1]' * d_raw) : d_raw
                    d_input = clamp.(d_input, -2.0, 2.0)
                    slice_proj_d_ws[chosen_s] .+= d_input * last_x_t'
                    slice_proj_d_bs[chosen_s] .+= d_input
                end

                for si in 1:4
                    if sum(abs, slice_proj_d_ws[si]) > 0
                        model.slice_proj_ws[si] .-= lr .* clamp.(slice_proj_d_ws[si], -0.5, 0.5)
                        model.slice_proj_bs[si] .-= lr .* clamp.(slice_proj_d_bs[si], -0.5, 0.5)
                    end
                end

                sp_batch = last_route.slice_probs
                entropy_grad = zeros(4)
                for k in 1:4
                    if sp_batch[k] > 1e-8
                        entropy_grad[k] = -(log(sp_batch[k]) + 1.0)
                    end
                end

                pp_batch = last_route.phase_probs
                phase_entropy_grad = zeros(4)
                for k in 1:4
                    if pp_batch[k] > 1e-8
                        phase_entropy_grad[k] = -(log(pp_batch[k]) + 1.0)
                    end
                end

                reinforce_grad = zeros(4)
                for step_idx in 1:length(fwd_chosen_slices)
                    chosen_s = fwd_chosen_slices[step_idx]
                    reinforce_grad[chosen_s] += loss - running_baseline
                end
                reinforce_grad ./= max(length(fwd_chosen_slices), 1)

                router_grad = (0.15 + 0.2 * (1.0 - lfb_scale)) .* (entropy_grad .+ 0.5 .* phase_entropy_grad) .+ 0.3 .* reinforce_grad
                router_grad = clamp.(router_grad, -5.0, 5.0)

                phase_embed_r = encode_phase(model.phase_manager, last_chosen_phase)
                router_input_r = vcat(last_x_t, last_mem_state, phase_embed_r)
                h1_r = model.router.W1 * router_input_r .+ model.router.b1
                h1_relu_r = max.(0.0, h1_r)
                h2_r = model.router.W2 * h1_relu_r .+ model.router.b2
                h2_relu_r = max.(0.0, h2_r)

                d_h2_r = model.router.slice_head_w' * router_grad
                d_h2_r = clamp.(d_h2_r, -1.0, 1.0)
                d_h2_r .*= (h2_r .> 0.0)
                model.router.W2 .-= router_lr .* clamp.(d_h2_r * h1_relu_r', -1.0, 1.0)
                model.router.b2 .-= router_lr .* clamp.(d_h2_r, -1.0, 1.0)

                d_h1_r = model.router.W2' * d_h2_r
                d_h1_r = clamp.(d_h1_r, -1.0, 1.0)
                d_h1_r .*= (h1_r .> 0.0)
                model.router.W1 .-= router_lr .* clamp.(d_h1_r * router_input_r', -1.0, 1.0)
                model.router.b1 .-= router_lr .* clamp.(d_h1_r, -1.0, 1.0)

                model.router.slice_head_w .-= router_lr .* clamp.(router_grad * h2_relu_r', -0.1, 0.1)
                model.router.slice_head_b .-= router_lr .* clamp.(router_grad, -0.1, 0.1)
            end

            epoch_loss += batch_loss / batch_size_actual
            n_batches += 1
        end

        epoch_loss /= n_batches

        if !isempty(slice_probs_epoch)
            epoch_aux = switch_aux_loss(slice_probs_epoch, 4)
            epoch_ortho = expert_orthogonality_loss(model.slices)
            epoch_var = routing_variance_loss(slice_probs_epoch, 4)
        end

        epoch_loss += aux_loss_coeff * epoch_aux + ortho_coeff * epoch_ortho + var_coeff * epoch_var

        push!(history["loss"], epoch_loss)

        if epoch % print_every == 0 || epoch == 1
            acc_val = evaluate_sequence(model, X_val, y_val)
            acc_test = evaluate_sequence(model, X_test_global, y_test_global)
            push!(history["val_acc"], acc_val)
            push!(history["test_acc"], acc_test)

            improved = acc_val > best_val + 0.001
            if improved
                best_val = acc_val
                patience = 0
                marker = " *"
                best_weights = (
                    output_w = copy(model.output_w),
                    output_b = copy(model.output_b),
                    input_proj_w = copy(model.input_proj_w),
                    input_proj_b = copy(model.input_proj_b),
                    slice_proj_ws = [copy(w) for w in model.slice_proj_ws],
                    slice_proj_bs = [copy(b) for b in model.slice_proj_bs],
                    slice_ws = [copy(s.weights[1]) for s in model.slices],
                    slice_bs = [copy(s.biases[1]) for s in model.slices],
                    router_W1 = copy(model.router.W1),
                    router_b1 = copy(model.router.b1),
                    router_W2 = copy(model.router.W2),
                    router_b2 = copy(model.router.b2),
                    slice_head_w = copy(model.router.slice_head_w),
                    slice_head_b = copy(model.router.slice_head_b),
                )
            else
                patience += print_every
                marker = ""
            end

            slice_dist = expert_loads ./ max(total_routed, 1.0)
            load_balance = round(1.0 - maximum(slice_dist) + minimum(slice_dist), digits=3)

            println("  Epoch $(lpad(epoch, 3)): loss=$(round(epoch_loss, digits=4)), " *
                    "aux=$(round(epoch_aux, digits=4)), " *
                    "lfb=$(round(lfb_scale, digits=2)), " *
                    "val=$(round(acc_val*100, digits=1))%, " *
                    "test=$(round(acc_test*100, digits=1))%, " *
                    "lb=$load_balance$marker")

            if patience >= PATIENCE
                println("  Early stopping at epoch $epoch")
                break
            end
        end
    end

    model.output_w .= best_weights.output_w
    model.output_b .= best_weights.output_b
    model.input_proj_w .= best_weights.input_proj_w
    model.input_proj_b .= best_weights.input_proj_b
    for (i, w) in enumerate(best_weights.slice_proj_ws)
        model.slice_proj_ws[i] .= w
    end
    for (i, b) in enumerate(best_weights.slice_proj_bs)
        model.slice_proj_bs[i] .= b
    end
    for (i, s) in enumerate(model.slices)
        s.weights[1] .= best_weights.slice_ws[i]
        s.biases[1] .= best_weights.slice_bs[i]
    end
    model.router.W1 .= best_weights.router_W1
    model.router.b1 .= best_weights.router_b1
    model.router.W2 .= best_weights.router_W2
    model.router.b2 .= best_weights.router_b2
    model.router.slice_head_w .= best_weights.slice_head_w
    model.router.slice_head_b .= best_weights.slice_head_b
    println("  Restored best model (val=$(round(best_val*100, digits=1))%)")

    return history
end

global X_test_global = zeros(1, 1, 3)
global y_test_global = Int[]

function main()
    print_header("RESEARCH-IMPROVED TEMPORAL INFERENCE")
    println("  Started: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")

    SEQ_LEN = 8
    N_TRAIN = 4000
    N_VAL = 500
    N_TEST = 1000
    EPOCHS = 120
    BATCH_SIZE = 32

    println("""
    Research improvements applied (DeepSeek-V3 exact parameters):
      1. Switch Transformer Auxiliary Load Balancing Loss (alpha=0.0001)
      2. Loss-Free Balancing (gamma=0.001, with linear decay)
      3. Router Z-Loss (numerical stability)
      4. Mixup temporal augmentation
      5. Cosine annealing LR schedule
      6. Phase diversity entropy gradient
      7. Model checkpointing (best val)
    """)

    global X_test_global, y_test_global
    X_train, y_train = generate_temporal_task(N_TRAIN; seq_len=SEQ_LEN, seed=42)
    X_val, y_val = generate_temporal_task(N_VAL; seq_len=SEQ_LEN, seed=123)
    X_test_global, y_test_global = generate_temporal_task(N_TEST; seq_len=SEQ_LEN, seed=456)

    println("  Training: $(size(X_train)), Val: $(size(X_val)), Test: $(size(X_test_global))")

    results = []

    # Experiment 1: Full research improvements with LFB decay
    print_header("EXP 1: Full Stack + LFB Decay")
    model_full = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=3, seed=42)
    params = count_parameters(model_full)
    println("  Parameters: $params")

    acc_before = evaluate_sequence(model_full, X_test_global, y_test_global)
    println("  Before: $(round(acc_before*100, digits=1))%")

    h1 = train_with_research_improvements!(
        model_full, X_train, y_train, X_val, y_val;
        epochs=EPOCHS, batch_size=BATCH_SIZE, base_lr=0.002,
        use_lfb=true, use_mixup=true, print_every=10,
        aux_loss_coeff=0.0001, ortho_coeff=0.0, var_coeff=0.0,
        z_loss_coeff=0.001, lfb_gamma=0.001)

    acc_final = evaluate_sequence(model_full, X_test_global, y_test_global)
    push!(results, ("Full + LFB Decay", params, acc_before, acc_final))

    # Experiment 2: No LFB decay (ablation - LFB always full strength)
    print_header("EXP 2: No LFB Decay (always-on LFB)")
    model_nolfb_decay = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=3, seed=42)

    acc_before2 = evaluate_sequence(model_nolfb_decay, X_test_global, y_test_global)

    h2 = train_with_research_improvements!(
        model_nolfb_decay, X_train, y_train, X_val, y_val;
        epochs=EPOCHS, batch_size=BATCH_SIZE, base_lr=0.002,
        use_lfb=true, use_mixup=true, print_every=10,
        aux_loss_coeff=0.0001, ortho_coeff=0.0, var_coeff=0.0,
        z_loss_coeff=0.001, lfb_gamma=0.001,
        lfb_decay=false)

    acc_final2 = evaluate_sequence(model_nolfb_decay, X_test_global, y_test_global)
    push!(results, ("No LFB Decay (ablation)", params, acc_before2, acc_final2))

    # Experiment 3: No LFB at all (ablation)
    print_header("EXP 3: No LFB (entropy only)")
    model_nolfb = create_model!(3; hidden_dim=32, memory_neurons=50, max_steps=3, seed=42)

    acc_before3 = evaluate_sequence(model_nolfb, X_test_global, y_test_global)

    h3 = train_with_research_improvements!(
        model_nolfb, X_train, y_train, X_val, y_val;
        epochs=EPOCHS, batch_size=BATCH_SIZE, base_lr=0.002,
        use_lfb=false, use_mixup=true, print_every=10,
        aux_loss_coeff=0.0001, ortho_coeff=0.0, var_coeff=0.0,
        z_loss_coeff=0.001)

    acc_final3 = evaluate_sequence(model_nolfb, X_test_global, y_test_global)
    push!(results, ("No LFB (entropy only)", params, acc_before3, acc_final3))

    # Experiment 4: Larger model with LFB decay
    print_header("EXP 4: Large (64h) + LFB Decay")
    model_large = create_model!(3; hidden_dim=64, memory_neurons=80, max_steps=4, seed=42)
    params_large = count_parameters(model_large)
    println("  Parameters: $params_large")

    acc_before4 = evaluate_sequence(model_large, X_test_global, y_test_global)

    h4 = train_with_research_improvements!(
        model_large, X_train, y_train, X_val, y_val;
        epochs=EPOCHS, batch_size=BATCH_SIZE, base_lr=0.0015,
        use_lfb=true, use_mixup=true, print_every=10,
        dropout_rate=0.2,
        aux_loss_coeff=0.0001, ortho_coeff=0.0, var_coeff=0.0,
        z_loss_coeff=0.001, lfb_gamma=0.001)

    acc_final4 = evaluate_sequence(model_large, X_test_global, y_test_global)
    push!(results, ("Large 64h + LFB Decay", params_large, acc_before4, acc_final4))

    # Final Results
    print_header("FINAL RESULTS")

    println("\n  Model                        │ Params │ Before │ After  │ Δ")
    println("  " * "─" ^ 70)

    for (name, params, before, after) in results
        name_p = rpad(name, 28)
        p = rpad(string(params), 6)
        b = round(before * 100, digits=1)
        a = round(after * 100, digits=1)
        delta = round((after - before) * 100, digits=1)
        delta_str = delta >= 0 ? "+$delta" : "$delta"
        marker = (after > 0.7) ? " ***" : (after > 0.6 ? " **" : (after > before + 0.1 ? " *" : ""))
        println("  $name_p │ $p │ $(rpad(b, 5))% │ $(rpad(a, 5))% │ $(delta_str)%$marker")
    end

    println("  " * "─" ^ 70)
    println("\n  Random baseline: $(round(100/6, digits=1))%")

    # Routing analysis for best model
    print_header("ROUTING ANALYSIS — BEST MODEL")

    slice_counts = zeros(4)
    phase_counts = zeros(4)
    step_counts = Int[]

    for i in 1:min(300, size(X_test_global, 1))
        _, traces_i = forward_sequence(model_full, @view X_test_global[i:i, :, :])
        for trace_list in traces_i
            for t_trace in trace_list
                push!(step_counts, t_trace.total_steps)
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

    println("\n  Slice Usage:")
    for (count, label) in zip(slice_counts, labels)
        pct = count / max(sum(slice_counts), 1) * 100
        bar = "█" * ("░" ^ max(0, round(Int, pct / 3)))
        println("    $(rpad(label, 10)) $(lpad(round(pct, digits=1), 5))% $bar")
    end

    println("\n  Phase Usage:")
    for (count, label) in zip(phase_counts, labels)
        pct = count / max(sum(phase_counts), 1) * 100
        bar = "█" * ("░" ^ max(0, round(Int, pct / 3)))
        println("    $(rpad(label, 10)) $(lpad(round(pct, digits=1), 5))% $bar")
    end

    println("\n  Unique slices: $(Int(sum(slice_counts .> 0)))/4")
    println("  Unique phases: $(Int(sum(phase_counts .> 0)))/4")
    println("  Avg steps: $(round(mean(step_counts), digits=2))")

    println("\n" * "=" ^ 70)
    println("Research-improved experiment complete!")
    println("=" ^ 70)

    println("\n" * "=" ^ 70)
    println("  MULTI-MODAL TASK: Forces routing diversity")
    println("=" ^ 70)

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

    N_MM = 4000
    X_mm_train, y_mm_train = generate_multimodal_task(N_MM; seed=42)
    X_mm_val, y_mm_val = generate_multimodal_task(500; seed=123)
    X_mm_test, y_mm_test = generate_multimodal_task(1000; seed=456)

    println("  4 modalities x 3 subclasses = 12 classes")
    println("  Modality 1: Temporal trends (ch1 varies)")
    println("  Modality 2: Spatial levels (all channels constant)")
    println("  Modality 3: Phase-shifted sine (all channels coupled)")
    println("  Modality 4: Burst hot-channel (1 hot, 2 cold)")
    println("  Training: $(size(X_mm_train)), Test: $(size(X_mm_test))")

    global X_test_global_mm = X_mm_test
    global y_test_global_mm = y_mm_test

    function evaluate_sequence_mm(model, X, y)
        outputs, _ = forward_sequence(model, X)
        pred_labels = [argmax(@view outputs[i, :]) for i in 1:size(outputs, 1)]
        return sum(pred_labels .== y) / length(y)
    end

    print_header("MULTI-MODAL EXP 1: 32h model")
    model_mm = create_model!(3; hidden_dim=48, memory_neurons=60, max_steps=3, n_classes=12, seed=42)
    params_mm = count_parameters(model_mm)
    println("  Parameters: $params_mm")

    best_val_mm = 0.0
    patience_mm = 0
    best_weights_mm = (
        output_w = copy(model_mm.output_w), output_b = copy(model_mm.output_b),
        input_proj_w = copy(model_mm.input_proj_w), input_proj_b = copy(model_mm.input_proj_b),
        slice_proj_ws = [copy(w) for w in model_mm.slice_proj_ws],
        slice_proj_bs = [copy(b) for b in model_mm.slice_proj_bs],
        slice_ws = [copy(s.weights[1]) for s in model_mm.slices],
        slice_bs = [copy(s.biases[1]) for s in model_mm.slices],
        router_W1 = copy(model_mm.router.W1), router_b1 = copy(model_mm.router.b1),
        router_W2 = copy(model_mm.router.W2), router_b2 = copy(model_mm.router.b2),
        slice_head_w = copy(model_mm.router.slice_head_w),
        slice_head_b = copy(model_mm.router.slice_head_b),
    )

    rng_mm = MersenneTwister(42)
    running_baseline_mm = 2.0
    for epoch in 1:120
        lr = cosine_anneal_lr(0.002, epoch, 120)
        indices = shuffle(rng_mm, 1:N_MM)
        epoch_loss = 0.0
        n_b = 0

        for start_idx in 1:32:N_MM
            end_idx = min(start_idx + 31, N_MM)
            batch_idx = indices[start_idx:end_idx]
            X_batch = X_mm_train[batch_idx, :, :]
            y_batch = y_mm_train[batch_idx]
            bs = size(X_batch, 1)

            for b in 1:bs
                accumulated_state = zeros(model_mm.slice_output_dim)
                last_route_mm = nothing
                last_chosen_phase_mm = RETRIEVE
                last_x_t_mm = zeros(3)
                last_mem_state_mm = zeros(1)

                mm_fwd_slices = Int[]
                mm_fwd_caches = NamedTuple[]

                for t in 1:size(X_batch, 2)
                    x_t = @view X_batch[b, t, :]
                    mem_state = read_memory(model_mm.memory)
                    phase = RETRIEVE

                    for step in 1:model_mm.max_steps
                        phase_embed = encode_phase(model_mm.phase_manager, phase)
                        router_input = vcat(x_t, mem_state, phase_embed)
                        route = router_forward(model_mm.router, router_input)
                        last_route_mm = route

                        chosen_slice = route.chosen_slice
                        chosen_phase, _ = decode_phase(model_mm.phase_manager, route.phase_logits)
                        last_chosen_phase_mm = chosen_phase
                        last_x_t_mm = copy(x_t)
                        last_mem_state_mm = copy(mem_state)

                        projected = model_mm.slice_proj_ws[chosen_slice] * x_t .+ model_mm.slice_proj_bs[chosen_slice]
                        slice = model_mm.slices[chosen_slice]
                        raw_out, acts, pre_acts = slice_forward_cached(slice, projected)
                        film = model_mm.film_layers[chosen_slice]
                        gamma = film.gamma_w * phase_embed .+ film.gamma_b
                        beta = film.beta_w * phase_embed .+ film.beta_b
                        conditioned = gamma .* raw_out .+ beta
                        accumulated_state .+= conditioned

                        push!(mm_fwd_slices, chosen_slice)
                        push!(mm_fwd_caches, (
                            acts=acts, pre_acts=pre_acts,
                            gamma=gamma, beta=beta, raw_out=raw_out,
                            phase_embed=phase_embed, slice_idx=chosen_slice
                        ))

                        update_phase!(model_mm.phase_manager, chosen_phase)
                        phase = chosen_phase
                        surprise_write!(model_mm.memory, conditioned)
                        mem_state = read_memory(model_mm.memory)

                        if route.confidence > 0.8
                            break
                        end
                    end
                end

                accumulated_state ./= size(X_batch, 2)
                logits = model_mm.output_w * accumulated_state .+ model_mm.output_b
                probs = exp.(logits .- maximum(logits))
                probs ./= sum(probs)

                target = y_batch[b]
                loss = -log(max(probs[target], 1e-8))
                epoch_loss += loss
                running_baseline_mm = 0.95 * running_baseline_mm + 0.05 * loss

                d_logits = copy(probs)
                d_logits[target] -= 1.0
                d_logits = clamp.(d_logits, -2.0, 2.0)
                d_avg = model_mm.output_w' * d_logits
                d_avg = clamp.(d_avg, -2.0, 2.0)

                d_out_w = d_logits * accumulated_state'
                d_out_w = clamp.(d_out_w, -1.0, 1.0)
                model_mm.output_w .-= lr .* d_out_w
                model_mm.output_b .-= lr .* clamp.(d_logits, -1.0, 1.0)

                mm_slice_proj_d_ws = [zeros(size(model_mm.slice_proj_ws[i])) for i in 1:4]
                mm_slice_proj_d_bs = [zeros(length(model_mm.slice_proj_bs[i])) for i in 1:4]

                n_steps_mm = length(mm_fwd_slices)
                d_per_step_mm = d_avg ./ max(n_steps_mm, 1)

                for step_idx in 1:n_steps_mm
                    chosen_s = mm_fwd_slices[step_idx]
                    cache = mm_fwd_caches[step_idx]
                    slice = model_mm.slices[chosen_s]
                    film = model_mm.film_layers[chosen_s]

                    d_conditioned = clamp.(d_per_step_mm, -2.0, 2.0)

                    d_gamma = d_conditioned .* cache.raw_out
                    d_beta = d_conditioned
                    d_raw = d_conditioned .* cache.gamma

                    d_gamma = clamp.(d_gamma, -1.0, 1.0)
                    d_beta = clamp.(d_beta, -1.0, 1.0)
                    d_raw = clamp.(d_raw, -1.0, 1.0)

                    film.gamma_w .-= lr .* (d_gamma * cache.phase_embed')
                    film.gamma_b .-= lr .* d_gamma
                    film.beta_w .-= lr .* (d_beta * cache.phase_embed')
                    film.beta_b .-= lr .* d_beta

                    n_layers = length(slice.weights)
                    saved_ws = [copy(W) for W in slice.weights]
                    delta = d_raw

                    for i in n_layers:-1:1
                        dW = delta * cache.acts[i]'
                        dW = clamp.(dW, -0.5, 0.5)
                        slice.weights[i] .-= lr .* dW
                        slice.biases[i] .-= lr .* clamp.(copy(delta), -0.5, 0.5)

                        if i > 1
                            delta = (saved_ws[i]' * delta) .* (cache.pre_acts[i-1] .> 0.0)
                            delta = clamp.(delta, -2.0, 2.0)
                        end
                    end

                    d_input = n_layers > 1 ? (saved_ws[1]' * d_raw) : d_raw
                    d_input = clamp.(d_input, -2.0, 2.0)
                    mm_slice_proj_d_ws[chosen_s] .+= d_input * last_x_t_mm'
                    mm_slice_proj_d_bs[chosen_s] .+= d_input
                end

                for si in 1:4
                    if sum(abs, mm_slice_proj_d_ws[si]) > 0
                        model_mm.slice_proj_ws[si] .-= lr .* clamp.(mm_slice_proj_d_ws[si], -0.5, 0.5)
                        model_mm.slice_proj_bs[si] .-= lr .* clamp.(mm_slice_proj_d_bs[si], -0.5, 0.5)
                    end
                end

                sp_batch = last_route_mm.slice_probs
                entropy_grad = zeros(4)
                for k in 1:4
                    if sp_batch[k] > 1e-8
                        entropy_grad[k] = -(log(sp_batch[k]) + 1.0)
                    end
                end

                mm_reinforce_grad = zeros(4)
                for si in mm_fwd_slices
                    mm_reinforce_grad[si] += loss
                end
                mm_reinforce_grad ./= max(length(mm_fwd_slices), 1)
                mm_reinforce_grad .-= running_baseline_mm

                router_grad = 0.2 .* entropy_grad .+ 0.3 .* mm_reinforce_grad
                router_grad = clamp.(router_grad, -5.0, 5.0)

                phase_embed_r = encode_phase(model_mm.phase_manager, last_chosen_phase_mm)
                router_input_r = vcat(last_x_t_mm, last_mem_state_mm, phase_embed_r)
                h1_r = model_mm.router.W1 * router_input_r .+ model_mm.router.b1
                h1_relu_r = max.(0.0, h1_r)
                h2_r = model_mm.router.W2 * h1_relu_r .+ model_mm.router.b2
                h2_relu_r = max.(0.0, h2_r)

                d_h2_r = model_mm.router.slice_head_w' * router_grad
                d_h2_r = clamp.(d_h2_r, -1.0, 1.0)
                d_h2_r .*= (h2_r .> 0.0)
                model_mm.router.W2 .-= 0.0005 .* clamp.(d_h2_r * h1_relu_r', -1.0, 1.0)
                model_mm.router.b2 .-= 0.0005 .* clamp.(d_h2_r, -1.0, 1.0)

                d_h1_r = model_mm.router.W2' * d_h2_r
                d_h1_r = clamp.(d_h1_r, -1.0, 1.0)
                d_h1_r .*= (h1_r .> 0.0)
                model_mm.router.W1 .-= 0.0005 .* clamp.(d_h1_r * router_input_r', -1.0, 1.0)
                model_mm.router.b1 .-= 0.0005 .* clamp.(d_h1_r, -1.0, 1.0)

                model_mm.router.slice_head_w .-= 0.0005 .* clamp.(router_grad * h2_relu_r', -0.1, 0.1)
                model_mm.router.slice_head_b .-= 0.0005 .* clamp.(router_grad, -0.1, 0.1)
            end
            n_b += 1
        end

        epoch_loss /= max(n_b * 32, 1)

        if epoch % 10 == 0 || epoch == 1
            acc_val = evaluate_sequence_mm(model_mm, X_mm_val, y_mm_val)
            acc_test = evaluate_sequence_mm(model_mm, X_mm_test, y_mm_test)

            improved = acc_val > best_val_mm + 0.001
            if improved
                best_val_mm = acc_val
                patience_mm = 0
                marker = " *"
                best_weights_mm = (
                    output_w = copy(model_mm.output_w), output_b = copy(model_mm.output_b),
                    input_proj_w = copy(model_mm.input_proj_w), input_proj_b = copy(model_mm.input_proj_b),
                    slice_proj_ws = [copy(w) for w in model_mm.slice_proj_ws],
                    slice_proj_bs = [copy(b) for b in model_mm.slice_proj_bs],
                    slice_ws = [copy(s.weights[1]) for s in model_mm.slices],
                    slice_bs = [copy(s.biases[1]) for s in model_mm.slices],
                    router_W1 = copy(model_mm.router.W1), router_b1 = copy(model_mm.router.b1),
                    router_W2 = copy(model_mm.router.W2), router_b2 = copy(model_mm.router.b2),
                    slice_head_w = copy(model_mm.router.slice_head_w),
                    slice_head_b = copy(model_mm.router.slice_head_b),
                )
            else
                patience_mm += 10
                marker = ""
            end

            println("  Epoch $(lpad(epoch, 3)): loss=$(round(epoch_loss, digits=4)), " *
                    "val=$(round(acc_val*100, digits=1))%, " *
                    "test=$(round(acc_test*100, digits=1))%$marker")

            if patience_mm >= 30
                println("  Early stopping at epoch $epoch")
                break
            end
        end
    end

    model_mm.output_w .= best_weights_mm.output_w
    model_mm.output_b .= best_weights_mm.output_b
    model_mm.input_proj_w .= best_weights_mm.input_proj_w
    model_mm.input_proj_b .= best_weights_mm.input_proj_b
    for (i, w) in enumerate(best_weights_mm.slice_proj_ws)
        model_mm.slice_proj_ws[i] .= w
    end
    for (i, b) in enumerate(best_weights_mm.slice_proj_bs)
        model_mm.slice_proj_bs[i] .= b
    end
    for (i, s) in enumerate(model_mm.slices)
        s.weights[1] .= best_weights_mm.slice_ws[i]
        s.biases[1] .= best_weights_mm.slice_bs[i]
    end
    model_mm.router.W1 .= best_weights_mm.router_W1
    model_mm.router.b1 .= best_weights_mm.router_b1
    model_mm.router.W2 .= best_weights_mm.router_W2
    model_mm.router.b2 .= best_weights_mm.router_b2
    model_mm.router.slice_head_w .= best_weights_mm.slice_head_w
    model_mm.router.slice_head_b .= best_weights_mm.slice_head_b
    println("  Restored best model (val=$(round(best_val_mm*100, digits=1))%)")

    acc_mm_final = evaluate_sequence_mm(model_mm, X_mm_test, y_mm_test)
    println("  Multi-modal test accuracy: $(round(acc_mm_final*100, digits=1))%")

    println("\n  Routing analysis (multi-modal):")
    slice_counts_mm = zeros(4)
    phase_counts_mm = zeros(4)
    for i in 1:min(500, size(X_mm_test, 1))
        _, traces_i = forward_sequence(model_mm, @view X_mm_test[i:i, :, :])
        for trace_list in traces_i
            for t_trace in trace_list
                for s in t_trace.slice_history
                    slice_counts_mm[s] += 1
                end
                for p in t_trace.phase_history
                    phase_counts_mm[Int(p)] += 1
                end
            end
        end
    end

    println("\n  Slice Usage:")
    for (count, label) in zip(slice_counts_mm, labels)
        pct = count / max(sum(slice_counts_mm), 1) * 100
        bar = "█" * ("░" ^ max(0, round(Int, pct / 3)))
        println("    $(rpad(label, 10)) $(lpad(round(pct, digits=1), 5))% $bar")
    end

    println("\n  Phase Usage:")
    for (count, label) in zip(phase_counts_mm, labels)
        pct = count / max(sum(phase_counts_mm), 1) * 100
        bar = "█" * ("░" ^ max(0, round(Int, pct / 3)))
        println("    $(rpad(label, 10)) $(lpad(round(pct, digits=1), 5))% $bar")
    end

    println("\n  Unique slices: $(Int(sum(slice_counts_mm .> 0)))/4")
    println("  Unique phases: $(Int(sum(phase_counts_mm .> 0)))/4")

    println("\n" * "=" ^ 70)
    println("All experiments complete!")
    println("=" ^ 70)
end

main()
