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

function main()
    println("=" ^ 70)
    println("  QUICK ROUTING DIVERSITY CHECK — PER-SLICE PROJECTIONS")
    println("=" ^ 70)

    SEQ_LEN = 8
    N_TRAIN = 2000
    N_VAL = 500
    EPOCHS = 40
    BATCH_SIZE = 32

    X_train, y_train = generate_multimodal_task(N_TRAIN; seed=42)
    X_val, y_val = generate_multimodal_task(N_VAL; seed=123)

    println("  4 modalities x 3 subclasses = 12 classes")
    println("  Training: $(size(X_train)), Val: $(size(X_val))")

    model = create_model!(3; hidden_dim=48, memory_neurons=60, max_steps=3, n_classes=12, seed=42)
    println("  Slice projection shapes: [$(join([size(w) for w in model.slice_proj_ws], ", "))]")

    rng = MersenneTwister(42)
    best_val = 0.0
    patience = 0
    running_baseline = 2.0

    for epoch in 1:EPOCHS
        lr = cosine_anneal_lr(0.002, epoch, EPOCHS)
        indices = shuffle(rng, 1:N_TRAIN)
        epoch_loss = 0.0
        n_b = 0

        expert_loads = zeros(4)

        for start_idx in 1:BATCH_SIZE:N_TRAIN
            end_idx = min(start_idx + BATCH_SIZE - 1, N_TRAIN)
            batch_idx = indices[start_idx:end_idx]
            X_batch = X_train[batch_idx, :, :]
            y_batch = y_train[batch_idx]
            bs = size(X_batch, 1)

            for b in 1:bs
                accumulated_state = zeros(model.slice_output_dim)
                last_route = nothing
                last_chosen_phase = RETRIEVE
                last_x_t = zeros(3)
                last_mem_state = zeros(1)

                fwd_slices = Int[]
                fwd_slice_caches = NamedTuple[]

                for t in 1:size(X_batch, 2)
                    x_t = @view X_batch[b, t, :]
                    mem_state = read_memory(model.memory)
                    phase = RETRIEVE

                    for step in 1:model.max_steps
                        phase_embed = encode_phase(model.phase_manager, phase)
                        router_input = vcat(x_t, mem_state, phase_embed)
                        route = router_forward(model.router, router_input)
                        last_route = route

                        chosen_slice = route.chosen_slice
                        chosen_phase, _ = decode_phase(model.phase_manager, route.phase_logits)
                        last_chosen_phase = chosen_phase
                        last_x_t = copy(x_t)
                        last_mem_state = copy(mem_state)

                        expert_loads[chosen_slice] += 1.0

                        projected = model.slice_proj_ws[chosen_slice] * x_t .+ model.slice_proj_bs[chosen_slice]
                        slice = model.slices[chosen_slice]
                        raw_out, acts, pre_acts = slice_forward_cached(slice, projected)
                        film = model.film_layers[chosen_slice]
                        gamma = film.gamma_w * phase_embed .+ film.gamma_b
                        beta = film.beta_w * phase_embed .+ film.beta_b
                        conditioned = gamma .* raw_out .+ beta
                        accumulated_state .+= conditioned

                        push!(fwd_slices, chosen_slice)
                        push!(fwd_slice_caches, (
                            acts=acts, pre_acts=pre_acts,
                            gamma=gamma, beta=beta, raw_out=raw_out,
                            phase_embed=phase_embed
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

                target = y_batch[b]
                loss = -log(max(probs[target], 1e-8))
                epoch_loss += loss
                running_baseline = 0.95 * running_baseline + 0.05 * loss

                d_logits = copy(probs)
                d_logits[target] -= 1.0
                d_logits = clamp.(d_logits, -2.0, 2.0)
                d_avg = model.output_w' * d_logits
                d_avg = clamp.(d_avg, -2.0, 2.0)

                d_out_w = d_logits * accumulated_state'
                d_out_w = clamp.(d_out_w, -1.0, 1.0)
                model.output_w .-= lr .* d_out_w
                model.output_b .-= lr .* clamp.(d_logits, -1.0, 1.0)

                n_steps = length(fwd_slices)
                d_per_step = d_avg ./ max(n_steps, 1)

                for step_idx in 1:n_steps
                    chosen_s = fwd_slices[step_idx]
                    cache = fwd_slice_caches[step_idx]
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
                        dW = clamp.(dW, -0.5, 0.5)
                        slice.weights[i] .-= lr .* dW
                        slice.biases[i] .-= lr .* clamp.(copy(delta), -0.5, 0.5)
                        if i > 1
                            delta = (saved_ws[i]' * delta) .* (cache.pre_acts[i-1] .> 0.0)
                            delta = clamp.(delta, -2.0, 2.0)
                        end
                    end
                end

                sp_batch = last_route.slice_probs
                entropy_grad = zeros(4)
                for k in 1:4
                    if sp_batch[k] > 1e-8
                        entropy_grad[k] = -(log(sp_batch[k]) + 1.0)
                    end
                end

                reinforce_grad = zeros(4)
                for si in fwd_slices
                    reinforce_grad[si] += loss - running_baseline
                end
                reinforce_grad ./= max(length(fwd_slices), 1)

                router_grad = 0.2 .* entropy_grad .+ 0.3 .* reinforce_grad
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
                model.router.W2 .-= 0.0005 .* clamp.(d_h2_r * h1_relu_r', -1.0, 1.0)
                model.router.b2 .-= 0.0005 .* clamp.(d_h2_r, -1.0, 1.0)

                d_h1_r = model.router.W2' * d_h2_r
                d_h1_r = clamp.(d_h1_r, -1.0, 1.0)
                d_h1_r .*= (h1_r .> 0.0)
                model.router.W1 .-= 0.0005 .* clamp.(d_h1_r * router_input_r', -1.0, 1.0)
                model.router.b1 .-= 0.0005 .* clamp.(d_h1_r, -1.0, 1.0)

                model.router.slice_head_w .-= 0.0005 .* clamp.(router_grad * h2_relu_r', -0.1, 0.1)
                model.router.slice_head_b .-= 0.0005 .* clamp.(router_grad, -0.1, 0.1)
            end
            n_b += 1
        end

        epoch_loss /= max(n_b * BATCH_SIZE, 1)

        if epoch % 5 == 0 || epoch == 1
            outputs, _ = forward_sequence(model, X_val)
            pred_labels = [argmax(@view outputs[i, :]) for i in 1:size(outputs, 1)]
            acc = sum(pred_labels .== y_val) / length(y_val)

            slice_dist = expert_loads ./ sum(expert_loads)
            balance = round(1.0 - (maximum(slice_dist) - minimum(slice_dist)), digits=3)

            println("  Epoch $(lpad(epoch, 3)): loss=$(round(epoch_loss, digits=4)), " *
                    "val=$(round(acc*100, digits=1))%, " *
                    "slice_dist=$([round(d*100, digits=1) for d in slice_dist]), " *
                    "balance=$balance")

            if acc > best_val + 0.001
                best_val = acc
                patience = 0
            else
                patience += 5
            end

            if patience >= 20
                println("  Early stopping at epoch $epoch")
                break
            end
        end

        expert_loads .*= 0.95
    end

    println("\n" * "=" ^ 70)
    println("  ROUTING ANALYSIS (FINAL)")
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
    println("  Best val accuracy: $(round(best_val*100, digits=1))%")

    println("\n  Projection similarity (cosine):")
    for i in 1:4
        for j in (i+1):4
            wi = vec(model.slice_proj_ws[i])
            wj = vec(model.slice_proj_ws[j])
            cos_sim = dot(wi, wj) / (norm(wi) * norm(wj) + 1e-8)
            println("    Slice $i vs $j: $(round(cos_sim, digits=4))")
        end
    end

    println("\n" * "=" ^ 70)
    println("  Quick routing check complete!")
    println("=" ^ 70)
end

main()
