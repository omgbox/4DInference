export train_with_backprop!, train_curriculum!

function softmax_stable(x::AbstractVector{Float64})
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    s = sum(exp_x)
    if s == 0.0 || !isfinite(s)
        return ones(length(x)) ./ length(x)
    end
    exp_x ./ s
    return exp_x
end

function slice_forward_cached(slice::Slice, x::AbstractVector{Float64})
    activations = Vector{Float64}[]
    pre_activations = Vector{Float64}[]

    h = copy(x)
    push!(activations, copy(h))

    for i in 1:length(slice.weights)
        z = slice.weights[i] * h .+ slice.biases[i]
        push!(pre_activations, copy(z))

        if i < length(slice.weights)
            h = max.(0.0, z)
        else
            h = z
        end
        push!(activations, copy(h))
    end

    return h, activations, pre_activations
end

function router_forward_with_cache(router::FourDRouter, x::AbstractVector{Float64})
    h1 = router.W1 * x .+ router.b1
    h1_relu = max.(0.0, h1)

    h2 = router.W2 * h1_relu .+ router.b2
    h2_relu = max.(0.0, h2)

    slice_logits = router.slice_head_w * h2_relu .+ router.slice_head_b
    slice_probs = softmax_stable(slice_logits)

    phase_logits = router.phase_head_w * h2_relu .+ router.phase_head_b
    phase_probs = softmax_stable(phase_logits)

    conf_raw = (router.confidence_head_w' * h2_relu + router.confidence_head_b)
    confidence = 1.0 / (1.0 + exp(-clamp(conf_raw, -10.0, 10.0)))

    return (
        slice_logits=slice_logits, slice_probs=slice_probs,
        phase_logits=phase_logits, phase_probs=phase_probs,
        confidence=confidence,
        h1=h1, h1_relu=h1_relu, h2=h2, h2_relu=h2_relu,
        x=x
    )
end

function compute_ranking_gradient!(router_scores::Vector{Float64}, losses::Vector{Float64})
    grad = zeros(length(router_scores))
    n = length(router_scores)
    
    for i in 1:n
        for j in 1:n
            if i != j && losses[i] < losses[j]
                dif = router_scores[i] - router_scores[j]
                dif_clamped = clamp(dif, -20.0, 20.0)
                # Gradient of log(1 + exp(-dif)) w.r.t. dif is -sigmoid(-dif)
                sigmoid_val = 1.0 / (1.0 + exp(clamp(-dif_clamped, -20.0, 20.0)))
                grad[i] -= sigmoid_val
                grad[j] += sigmoid_val
            end
        end
    end
    
    return grad
end

function compute_slice_loss_only(model::FourDModel, x::AbstractVector{Float64}, routed_slice::Int, target::Int)
    mem_state = read_memory(model.memory)
    phase = RETRIEVE
    combined = zeros(model.combined_output_dim)
    
    for step in 1:model.max_steps
        phase_embed = encode_phase(model.phase_manager, phase)
        
        compress_proj = model.slice_proj_ws[COMPRESS_IDX] * x .+ model.slice_proj_bs[COMPRESS_IDX]
        compress_out, _, _ = slice_forward_cached(model.slices[COMPRESS_IDX], compress_proj)
        compress_film = model.film_layers[COMPRESS_IDX]
        compress_gamma = compress_film.gamma_w * phase_embed .+ compress_film.gamma_b
        compress_beta = compress_film.beta_w * phase_embed .+ compress_film.beta_b
        compress_cond = compress_gamma .* compress_out .+ compress_beta
        
        routed_proj = model.slice_proj_ws[routed_slice] * x .+ model.slice_proj_bs[routed_slice]
        routed_out, _, _ = slice_forward_cached(model.slices[routed_slice], routed_proj)
        routed_film = model.film_layers[routed_slice]
        routed_gamma = routed_film.gamma_w * phase_embed .+ routed_film.gamma_b
        routed_beta = routed_film.beta_w * phase_embed .+ routed_film.beta_b
        routed_cond = routed_gamma .* routed_out .+ routed_beta
        
        combined = vcat(compress_cond, routed_cond)
    end
    
    logits = model.output_w * combined .+ model.output_b
    
    probs = softmax_stable(logits)
    clamp!(probs, 1e-8, 1.0)
    probs ./= sum(probs)
    loss = -log(clamp(probs[target], 1e-15, 1.0))
    
    if !isfinite(loss)
        loss = 10.0
    end
    
    return loss
end

function forward_all_routed_fast(model::FourDModel, x::AbstractVector{Float64}, target::Int)
    saved_memory = copy(model.memory.state)
    saved_phase = model.phase_manager.current
    
    losses = zeros(N_ROUTED_SLICES)
    
    for k in 1:N_ROUTED_SLICES
        model.memory.state .= 0.0
        model.phase_manager.current = RETRIEVE
        
        losses[k] = compute_slice_loss_only(model, x, k, target)
    end
    
    model.memory.state .= saved_memory
    model.phase_manager.current = saved_phase
    
    return losses
end

function forward_single(model::FourDModel, x::AbstractVector{Float64}; force_routed::Union{Int,Nothing}=nothing)
    mem_state = read_memory(model.memory)

    phase = RETRIEVE
    step_outputs = Vector{Float64}[]
    chosen_slices = Int[]
    chosen_phases = PhaseType[]
    slice_probs_list = Vector{Float64}[]
    compress_caches = NamedTuple[]
    routed_caches = NamedTuple[]
    mem_states = Vector{Float64}[]

    for step in 1:model.max_steps
        phase_embed = encode_phase(model.phase_manager, phase)
        router_input = vcat(x, mem_state, phase_embed)
        r = router_forward(model.router, router_input)

        routed_slice = force_routed !== nothing ? force_routed : r.chosen_slice
        chosen_phase, _ = decode_phase(model.phase_manager, r.phase_logits)

        compress_proj = model.slice_proj_ws[COMPRESS_IDX] * x .+ model.slice_proj_bs[COMPRESS_IDX]
        compress_out, compress_acts, compress_pre_acts = slice_forward_cached(model.slices[COMPRESS_IDX], compress_proj)
        compress_film = model.film_layers[COMPRESS_IDX]
        compress_gamma = compress_film.gamma_w * phase_embed .+ compress_film.gamma_b
        compress_beta = compress_film.beta_w * phase_embed .+ compress_film.beta_b
        compress_cond = compress_gamma .* compress_out .+ compress_beta

        routed_proj = model.slice_proj_ws[routed_slice] * x .+ model.slice_proj_bs[routed_slice]
        routed_out, routed_acts, routed_pre_acts = slice_forward_cached(model.slices[routed_slice], routed_proj)
        routed_film = model.film_layers[routed_slice]
        routed_gamma = routed_film.gamma_w * phase_embed .+ routed_film.gamma_b
        routed_beta = routed_film.beta_w * phase_embed .+ routed_film.beta_b
        routed_cond = routed_gamma .* routed_out .+ routed_beta

        combined = vcat(compress_cond, routed_cond)

        push!(step_outputs, copy(combined))
        push!(chosen_slices, routed_slice)
        push!(chosen_phases, chosen_phase)
        push!(slice_probs_list, copy(r.slice_probs))
        push!(compress_caches, (
            acts=compress_acts, pre_acts=compress_pre_acts,
            gamma=compress_gamma, beta=compress_beta, raw_out=compress_out,
            phase_embed=phase_embed, slice_idx=COMPRESS_IDX,
            projected=copy(compress_proj)
        ))
        push!(routed_caches, (
            acts=routed_acts, pre_acts=routed_pre_acts,
            gamma=routed_gamma, beta=routed_beta, raw_out=routed_out,
            phase_embed=phase_embed, slice_idx=routed_slice,
            projected=copy(routed_proj)
        ))
        push!(mem_states, copy(mem_state))

        update_phase!(model.phase_manager, chosen_phase)
        phase = chosen_phase

        surprise_write!(model.memory, compress_cond)
        mem_state = read_memory(model.memory)

        if r.confidence > 0.8
            break
        end
    end

    n_steps = length(step_outputs)
    avg_output = zeros(model.combined_output_dim)
    for o in step_outputs
        avg_output .+= o
    end
    avg_output ./= n_steps

    logits = model.output_w * avg_output .+ model.output_b

    return (
        logits=logits,
        step_outputs=step_outputs,
        chosen_slices=chosen_slices,
        chosen_phases=chosen_phases,
        slice_probs=slice_probs_list,
        compress_caches=compress_caches,
        routed_caches=routed_caches,
        mem_states=mem_states,
        n_steps=n_steps,
        avg_output=avg_output
    )
end

function _backprop_through_slice!(slice, film, cache, d_per_step, x,
                                  slice_proj_grads_w, slice_proj_grads_b, lr)
    si = cache.slice_idx
    d_conditioned = d_per_step
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
    deltas = Vector{Vector{Float64}}(undef, n_layers)
    delta = d_raw

    for i in n_layers:-1:1
        deltas[i] = delta
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

    d_input = n_layers > 1 ? (saved_ws[1]' * deltas[1]) : deltas[1]
    d_input = clamp.(d_input, -2.0, 2.0)
    slice_proj_grads_w[si] .+= d_input * x'
    slice_proj_grads_b[si] .+= d_input
end

function train_with_backprop!(model::FourDModel, train_data::DataLoader,
                              val_X::AbstractMatrix{Float64},
                              val_y::AbstractVector{Int},
                              val_diff::AbstractVector{Int};
                              epochs::Int=100,
                              lr::Float64=0.001,
                              router_lr::Float64=0.0005,
                              ensemble_mode::Bool=true,
                              print_every::Int=10)

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

            batch_slice_counts = zeros(N_ROUTED_SLICES)

            for sample_idx in 1:size(X_batch, 1)
                x = @view X_batch[sample_idx, :]
                target = y_batch[sample_idx]
                difficulty = diff_batch[sample_idx]

                fwd = forward_single(model, x)

                probs = softmax_stable(fwd.logits)
                clamp!(probs, 1e-8, 1.0)
                probs ./= sum(probs)
                loss = -log(probs[target])

                if !isfinite(loss)
                    continue
                end

                d_logits = copy(probs)
                d_logits[target] -= 1.0
                d_logits = clamp.(d_logits, -2.0, 2.0)

                d_avg = model.output_w' * d_logits
                d_avg = clamp.(d_avg, -2.0, 2.0)

                d_compress_avg = @view d_avg[1:model.slice_output_dim]
                d_routed_avg = @view d_avg[(model.slice_output_dim + 1):end]

                d_per_step_compress = d_compress_avg ./ fwd.n_steps
                d_per_step_routed = d_routed_avg ./ fwd.n_steps

                d_out_w = d_logits * fwd.avg_output'
                d_out_w = clamp.(d_out_w, -1.0, 1.0)
                model.output_w .-= lr .* d_out_w
                model.output_b .-= lr .* clamp.(d_logits, -1.0, 1.0)

                slice_proj_grads_w = [zeros(size(model.slice_proj_ws[i])) for i in 1:N_SLICES]
                slice_proj_grads_b = [zeros(length(model.slice_proj_bs[i])) for i in 1:N_SLICES]

                for step_idx in 1:fwd.n_steps
                    cc = fwd.compress_caches[step_idx]
                    _backprop_through_slice!(
                        model.slices[COMPRESS_IDX], model.film_layers[COMPRESS_IDX],
                        cc, d_per_step_compress, x,
                        slice_proj_grads_w, slice_proj_grads_b, lr
                    )

                    rc = fwd.routed_caches[step_idx]
                    _backprop_through_slice!(
                        model.slices[rc.slice_idx], model.film_layers[rc.slice_idx],
                        rc, d_per_step_routed, x,
                        slice_proj_grads_w, slice_proj_grads_b, lr
                    )
                end

                for si in 1:N_SLICES
                    if sum(abs, slice_proj_grads_w[si]) > 0
                        slice_proj_grads_w[si] = clamp.(slice_proj_grads_w[si], -1.0, 1.0)
                        slice_proj_grads_b[si] = clamp.(slice_proj_grads_b[si], -1.0, 1.0)
                        model.slice_proj_ws[si] .-= lr .* slice_proj_grads_w[si]
                        model.slice_proj_bs[si] .-= lr .* slice_proj_grads_b[si]
                    end
                end

                # Ranking loss: compute per-slice losses and train router on rankings
                slice_losses = forward_all_routed_fast(model, x, target)
                
                for step_idx in 1:fwd.n_steps
                    phase_embed = encode_phase(model.phase_manager, RETRIEVE)
                    router_input = vcat(x, fwd.mem_states[step_idx], phase_embed)
                    r_cache = router_forward_with_cache(model.router, router_input)
                    
                    # Get router scores for all 3 slices
                    router_scores = r_cache.slice_logits[1:N_ROUTED_SLICES]
                    
                    # Compute ranking gradients
                    ranking_grad = compute_ranking_gradient!(router_scores, slice_losses)
                    ranking_grad = clamp.(ranking_grad, -5.0, 5.0)
                    
                    # Backprop through router (only for routed slices, not COMPRESS)
                    d_h2 = zeros(length(r_cache.h2_relu))
                    d_h2[1:N_ROUTED_SLICES] .= ranking_grad
                    d_h2 = clamp.(d_h2, -1.0, 1.0)
                    d_h2 .*= (r_cache.h2 .> 0.0)
                    
                    model.router.W2 .-= router_lr .* clamp.(d_h2 * r_cache.h1_relu', -1.0, 1.0)
                    model.router.b2 .-= router_lr .* clamp.(d_h2, -1.0, 1.0)
                    
                    d_h1 = model.router.W2' * d_h2
                    d_h1 = clamp.(d_h1, -1.0, 1.0)
                    d_h1 .*= (r_cache.h1 .> 0.0)
                    
                    model.router.W1 .-= router_lr .* clamp.(d_h1 * router_input', -1.0, 1.0)
                    model.router.b1 .-= router_lr .* clamp.(d_h1, -1.0, 1.0)
                    
                    model.router.slice_head_w .-= router_lr .* clamp.(ranking_grad * r_cache.h2_relu', -0.1, 0.1)
                    model.router.slice_head_b .-= router_lr .* clamp.(ranking_grad, -0.1, 0.1)
                end

                epoch_loss += loss
            end

            n_batches += 1
        end

        epoch_loss /= max(n_batches, 1)
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

function train_curriculum!(model::FourDModel, n_samples::Int;
                           epochs_per_phase::Int=20,
                           lr::Float64=0.001,
                           router_lr::Float64=0.0005,
                           seed::Int=42)

    println("\n" * "=" ^ 60)
    println("CURRICULUM TRAINING")
    println("=" ^ 60)

    println("\nPhase 1: Train on EASY samples only (difficulty=1)")
    println("-" ^ 60)
    X_easy, y_easy, diff_easy = generate_dataset(n_samples; difficulty=1, seed=seed)
    X_val, y_val, diff_val = generate_dataset(div(n_samples, 5); seed=seed + 1000)
    train_loader = DataLoader(X_easy, y_easy, diff_easy; batch_size=64)
     h1 = train_with_backprop!(model, train_loader, X_val, y_val, diff_val;
                               epochs=epochs_per_phase, lr=lr, router_lr=router_lr,
                               print_every=5)

    println("\nPhase 2: Train on MEDIUM samples (difficulty=2)")
    println("-" ^ 60)
    X_med, y_med, diff_med = generate_dataset(n_samples; difficulty=2, seed=seed + 1)
    X_val2, y_val2, diff_val2 = generate_dataset(div(n_samples, 5); seed=seed + 1001)
    train_loader2 = DataLoader(X_med, y_med, diff_med; batch_size=64)
     h2 = train_with_backprop!(model, train_loader2, X_val2, y_val2, diff_val2;
                               epochs=epochs_per_phase, lr=lr * 0.5, router_lr=router_lr * 0.5,
                               print_every=5)

    println("\nPhase 3: Train on HARD samples (difficulty=3)")
    println("-" ^ 60)
    X_hard, y_hard, diff_hard = generate_dataset(n_samples; difficulty=3, seed=seed + 2)
    X_val3, y_val3, diff_val3 = generate_dataset(div(n_samples, 5); seed=seed + 1002)
    train_loader3 = DataLoader(X_hard, y_hard, diff_hard; batch_size=64)
     h3 = train_with_backprop!(model, train_loader3, X_val3, y_val3, diff_val3;
                               epochs=epochs_per_phase, lr=lr * 0.25, router_lr=router_lr * 0.25,
                               print_every=5)

    println("\nPhase 4: Train on ALL difficulties (mixed)")
    println("-" ^ 60)
    X_all, y_all, diff_all = generate_dataset(n_samples * 3; seed=seed + 3)
    X_val4, y_val4, diff_val4 = generate_dataset(div(n_samples, 5); seed=seed + 1003)
    train_loader4 = DataLoader(X_all, y_all, diff_all; batch_size=64)
     h4 = train_with_backprop!(model, train_loader4, X_val4, y_val4, diff_val4;
                               epochs=epochs_per_phase * 2, lr=lr * 0.1, router_lr=router_lr * 0.1,
                               print_every=5)

    combined_history = Dict{String, Vector{Float64}}(
        "loss" => vcat(h1["loss"], h2["loss"], h3["loss"], h4["loss"]),
        "accuracy" => vcat(h1["accuracy"], h2["accuracy"], h3["accuracy"], h4["accuracy"]),
        "easy_steps" => vcat(h1["easy_steps"], h2["easy_steps"], h3["easy_steps"], h4["easy_steps"]),
        "hard_steps" => vcat(h1["hard_steps"], h2["hard_steps"], h3["hard_steps"], h4["hard_steps"])
    )

    println("\n" * "=" ^ 60)
    println("Curriculum training complete!")
    println("=" ^ 60)

    return combined_history
end
