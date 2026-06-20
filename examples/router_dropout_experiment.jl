using Random
using Statistics
using Dates

include(joinpath(@__DIR__, "..", "src", "FourDInference.jl"))
using .FourDInference

function print_header(title::String)
    width = 70
    println("\n" * "=" ^ width)
    println(" " ^ max(0, div(width - length(title), 2)) * title)
    println("=" ^ width)
end

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

function evaluate_sequence(model::FourDModel, X::AbstractArray{Float64, 3},
                           y::AbstractVector{Int})
    outputs, _ = forward_sequence(model, X)
    pred_labels = [argmax(@view outputs[i, :]) for i in 1:size(outputs, 1)]
    return sum(pred_labels .== y) / length(y)
end

function cosine_anneal_lr(base_lr::Float64, epoch::Int, max_epochs::Int)
    return base_lr * 0.5 * (1.0 + cos(π * epoch / max_epochs))
end

function count_parameters(model::FourDModel)
    total = 0
    total += length(model.input_proj_w) + length(model.input_proj_b)
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

function softmax_stable(x::AbstractVector{Float64})
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    s = sum(exp_x)
    if s == 0.0 || !isfinite(s)
        return ones(length(x)) ./ length(x)
    end
    return exp_x ./ s
end

function train_with_router_dropout!(model::FourDModel, X_train, y_train, X_val, y_val;
                                     epochs::Int=120, batch_size::Int=32,
                                     base_lr::Float64=0.002, router_lr::Float64=0.0005,
                                     router_dropout::Float64=0.0,
                                     print_every::Int=10)

    n_train = size(X_train, 1)
    rng = MersenneTwister(42)
    best_val = 0.0
    patience = 0
    PATIENCE = 30

    best_weights = save_weights(model)
    slice_counts = zeros(4)
    total_routed = 0.0

    for epoch in 1:epochs
        lr = cosine_anneal_lr(base_lr, epoch, epochs)
        indices = shuffle(rng, 1:n_train)
        epoch_loss = 0.0
        n_b = 0

        for start_idx in 1:batch_size:n_train
            end_idx = min(start_idx + batch_size - 1, n_train)
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

                for t in 1:size(X_batch, 2)
                    x_t = @view X_batch[b, t, :]
                    projected = model.input_proj_w * x_t .+ model.input_proj_b
                    mem_state = read_memory(model.memory)
                    phase = RETRIEVE

                    for step in 1:model.max_steps
                        phase_embed = encode_phase(model.phase_manager, phase)
                        router_input = vcat(x_t, mem_state, phase_embed)
                        route = router_forward(model.router, router_input)
                        last_route = route

                        chosen_slice = route.chosen_slice
                        chosen_phase, _ = decode_phase(model.phase_manager, route.phase_logits)

                        if router_dropout > 0.0 && rand(rng) < router_dropout
                            other_slices = [s for s in 1:4 if s != chosen_slice]
                            if !isempty(other_slices)
                                chosen_slice = rand(rng, other_slices)
                            end
                        end

                        last_chosen_phase = chosen_phase
                        last_x_t = copy(x_t)
                        last_mem_state = copy(mem_state)

                        slice_out = model.slices[chosen_slice]
                        raw_output = slice_forward(slice_out, projected)
                        film = model.film_layers[chosen_slice]
                        conditioned = film_forward(film, raw_output, phase_embed)
                        accumulated_state .+= conditioned

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

                d_logits = copy(probs)
                d_logits[target] -= 1.0
                d_logits = clamp.(d_logits, -2.0, 2.0)
                d_avg = model.output_w' * d_logits
                d_avg = clamp.(d_avg, -2.0, 2.0)

                d_out_w = d_logits * accumulated_state'
                d_out_w = clamp.(d_out_w, -1.0, 1.0)
                model.output_w .-= lr .* d_out_w
                model.output_b .-= lr .* clamp.(d_logits, -1.0, 1.0)

                for t in 1:size(X_batch, 2)
                    x_t = @view X_batch[b, t, :]
                    d_input = d_avg ./ size(X_batch, 2)
                    d_input = clamp.(d_input, -2.0, 2.0)
                    d_proj = d_input * x_t'
                    d_proj = clamp.(d_proj, -0.5, 0.5)
                    model.input_proj_w .-= lr .* d_proj
                    model.input_proj_b .-= lr .* clamp.(d_input, -0.5, 0.5)
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

                router_grad = 0.2 .* (entropy_grad .+ 0.5 .* phase_entropy_grad)
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
            n_b += 1
        end

        epoch_loss /= max(n_b * batch_size, 1)

        if epoch % print_every == 0 || epoch == 1
            acc_val = evaluate_sequence(model, X_val, y_val)

            improved = acc_val > best_val + 0.001
            if improved
                best_val = acc_val
                patience = 0
                best_weights = save_weights(model)
                marker = " *"
            else
                patience += print_every
                marker = ""
            end

            println("  Epoch $(lpad(epoch, 3)): loss=$(round(epoch_loss, digits=4)), " *
                    "val=$(round(acc_val*100, digits=1))%$marker")

            if patience >= PATIENCE
                println("  Early stopping at epoch $epoch")
                break
            end
        end
    end

    load_weights!(model, best_weights)
    return best_val
end

function save_weights(model::FourDModel)
    return (
        output_w = copy(model.output_w), output_b = copy(model.output_b),
        input_proj_w = copy(model.input_proj_w), input_proj_b = copy(model.input_proj_b),
        slice_ws = [copy(s.weights[1]) for s in model.slices],
        slice_bs = [copy(s.biases[1]) for s in model.slices],
        router_W1 = copy(model.router.W1), router_b1 = copy(model.router.b1),
        router_W2 = copy(model.router.W2), router_b2 = copy(model.router.b2),
        slice_head_w = copy(model.router.slice_head_w),
        slice_head_b = copy(model.router.slice_head_b),
    )
end

function load_weights!(model::FourDModel, w)
    model.output_w .= w.output_w
    model.output_b .= w.output_b
    model.input_proj_w .= w.input_proj_w
    model.input_proj_b .= w.input_proj_b
    for (i, s) in enumerate(model.slices)
        s.weights[1] .= w.slice_ws[i]
        s.biases[1] .= w.slice_bs[i]
    end
    model.router.W1 .= w.router_W1
    model.router.b1 .= w.router_b1
    model.router.W2 .= w.router_W2
    model.router.b2 .= w.router_b2
    model.router.slice_head_w .= w.slice_head_w
    model.router.slice_head_b .= w.slice_head_b
end

function analyze_routing(model::FourDModel, X, y; max_samples::Int=500)
    slice_counts = zeros(4)
    phase_counts = zeros(4)
    for i in 1:min(max_samples, size(X, 1))
        _, traces_i = forward_sequence(model, @view X[i:i, :, :])
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

    n_unique = Int(sum(slice_counts .> 0))
    println("  Unique slices: $n_unique/4")
    return n_unique
end

function main()
    print_header("ROUTER DROPOUT EXPERIMENT")
    println("  Testing if randomly disabling slices during training forces diversity")

    X_train, y_train = generate_multimodal_task(4000; seed=42)
    X_val, y_val = generate_multimodal_task(500; seed=123)
    X_test, y_test = generate_multimodal_task(1000; seed=456)

    println("  Task: 4 modalities x 3 subclasses = 12 classes")
    println("  Training: $(size(X_train)), Test: $(size(X_test))")

    results = []

    dropout_rates = [0.0, 0.15, 0.25, 0.35, 0.5]

    for dr in dropout_rates
        print_header("Router dropout = $(round(dr * 100, digits=0))%")

        model = create_model!(3; hidden_dim=48, memory_neurons=60, max_steps=3, n_classes=12, seed=42)
        params = count_parameters(model)
        println("  Parameters: $params")

        best_val = train_with_router_dropout!(
            model, X_train, y_train, X_val, y_val;
            epochs=120, batch_size=32, base_lr=0.002, router_lr=0.0005,
            router_dropout=dr, print_every=20)

        acc_test = evaluate_sequence(model, X_test, y_test)
        n_unique = analyze_routing(model, X_test, y_test)
        push!(results, (dr, params, best_val, acc_test, n_unique))

        println("  Best val: $(round(best_val*100, digits=1))%, Test: $(round(acc_test*100, digits=1))%, Unique slices: $n_unique/4")
    end

    print_header("FINAL RESULTS")
    println("\n  Dropout │ Params │  Val  │  Test │ Slices")
    println("  " * "─" ^ 50)
    for (dr, params, val, test, slices) in results
        dr_str = rpad("$(round(dr*100, digits=0))%", 8)
        p_str = rpad(string(params), 6)
        v_str = rpad("$(round(val*100, digits=1))%", 6)
        t_str = rpad("$(round(test*100, digits=1))%", 6)
        marker = slices > 1 ? " ✓" : ""
        println("  $dr_str │ $p_str │ $v_str │ $t_str │ $slices/4$marker")
    end
    println("  " * "─" ^ 50)
    println("\n  Random baseline for 12 classes: $(round(100/12, digits=1))%")
    println("\n  Key insight: Higher dropout forces more slices to be used,")
    println("  but may reduce accuracy if slices are not specialized.")
    println("  The sweet spot is where routing diversity AND accuracy are both high.")

    println("\n" * "=" ^ 70)
    println("Router dropout experiment complete!")
    println("=" ^ 70)
end

main()
