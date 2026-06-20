export FourDModel, create_model!, forward, InferenceTrace, COMPRESS_IDX

using Random
using LinearAlgebra

const COMPRESS_IDX = 4

struct InferenceStep
    slice_idx::Int
    slice_name::String
    phase::PhaseType
    confidence::Float64
    memory_surprise::Float64
    output::Vector{Float64}
end

struct InferenceTrace
    steps::Vector{InferenceStep}
    final_output::Vector{Float64}
    total_steps::Int
    phase_history::Vector{PhaseType}
    slice_history::Vector{Int}
end

mutable struct FourDModel
    router::FourDRouter
    slices::Vector{Slice}
    memory::SORNMemory
    phase_manager::PhaseManager
    film_layers::Vector{FilmLayer}
    input_proj_w::Matrix{Float64}
    input_proj_b::Vector{Float64}
    slice_proj_ws::Vector{Matrix{Float64}}
    slice_proj_bs::Vector{Vector{Float64}}
    output_w::Matrix{Float64}
    output_b::Vector{Float64}
    input_dim::Int
    n_classes::Int
    memory_summary_dim::Int
    phase_embed_dim::Int
    slice_output_dim::Int
    combined_output_dim::Int
    max_steps::Int
    rng::MersenneTwister
end

function create_model!(input_dim::Int=3; hidden_dim::Int=16, memory_neurons::Int=10,
                       max_steps::Int=5, n_classes::Int=6, seed::Union{Int,Nothing}=nothing)

    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()

    memory_summary_dim = memory_neurons
    phase_embed_dim = 8
    slice_output_dim = 8
    combined_output_dim = slice_output_dim * 2

    router_input_dim = input_dim + memory_summary_dim + phase_embed_dim

    router = create_router!(router_input_dim, hidden_dim; seed=rand(rng, 1:10000))

    slices = create_slices!(slice_output_dim, hidden_dim, slice_output_dim; seed=rand(rng, 1:10000))

    memory = create_sorn_memory!(input_dim; n_neurons=memory_neurons,
                                  seed=rand(rng, 1:10000))

    phase_mgr = PhaseManager(; seed=rand(rng, 1:10000))

    film_layers = FilmLayer[]
    for i in 1:N_SLICES
        push!(film_layers, FilmLayer(phase_embed_dim, slice_output_dim;
                                     seed=rand(rng, 1:10000)))
    end

    output_w = randn(rng, n_classes, combined_output_dim) .* 0.01
    output_b = zeros(n_classes)

    input_proj_w = randn(rng, slice_output_dim, input_dim) .* sqrt(2.0 / input_dim)
    input_proj_b = zeros(slice_output_dim)

    slice_proj_ws = Matrix{Float64}[]
    slice_proj_bs = Vector{Float64}[]
    for i in 1:N_SLICES
        push!(slice_proj_ws, randn(rng, slice_output_dim, input_dim) .* sqrt(2.0 / input_dim))
        push!(slice_proj_bs, zeros(slice_output_dim))
    end

    return FourDModel(
        router, slices, memory, phase_mgr, film_layers,
        input_proj_w, input_proj_b,
        slice_proj_ws, slice_proj_bs,
        output_w, output_b,
        input_dim, n_classes, memory_summary_dim, phase_embed_dim,
        slice_output_dim, combined_output_dim,
        max_steps, rng
    )
end

function forward(model::FourDModel, x::AbstractVector{Float64};
                 verbose::Bool=false)
    steps = InferenceStep[]
    slice_history = Int[]
    phase_history = PhaseType[]

    phase = RETRIEVE

    for step in 1:model.max_steps
        mem_state = read_memory(model.memory)
        phase_embed = encode_phase(model.phase_manager, phase)

        router_input = vcat(x, mem_state, phase_embed)

        route = router_forward(model.router, router_input)

        chosen_phase, _ = decode_phase(model.phase_manager, route.phase_logits)
        update_phase!(model.phase_manager, chosen_phase)
        phase = chosen_phase

        routed_slice_idx = route.chosen_slice

        compress_proj = model.slice_proj_ws[COMPRESS_IDX] * x .+ model.slice_proj_bs[COMPRESS_IDX]
        compress_out = slice_forward(model.slices[COMPRESS_IDX], compress_proj)
        compress_cond = film_forward(model.film_layers[COMPRESS_IDX], compress_out, phase_embed)

        routed_proj = model.slice_proj_ws[routed_slice_idx] * x .+ model.slice_proj_bs[routed_slice_idx]
        routed_out = slice_forward(model.slices[routed_slice_idx], routed_proj)
        routed_cond = film_forward(model.film_layers[routed_slice_idx], routed_out, phase_embed)

        combined = vcat(compress_cond, routed_cond)

        old_state = copy(model.memory.state)
        surprise_write!(model.memory, compress_cond)
        memory_surprise = norm(model.memory.state - old_state)

        push!(steps, InferenceStep(
            routed_slice_idx,
            model.slices[routed_slice_idx].name,
            phase,
            route.confidence,
            memory_surprise,
            copy(combined)
        ))
        push!(slice_history, routed_slice_idx)
        push!(phase_history, phase)

        verbose && println("  Step $step: slice=$(model.slices[routed_slice_idx].name), phase=$phase, conf=$(round(route.confidence, digits=3))")

        if route.confidence > 0.8
            break
        end
    end

    last_output = isempty(steps) ? zeros(model.combined_output_dim) : steps[end].output
    final_output = model.output_w * last_output .+ model.output_b

    return InferenceTrace(
        steps, final_output, length(steps),
        phase_history, slice_history
    )
end

function forward(model::FourDModel, X::AbstractMatrix{Float64};
                 verbose::Bool=false)
    batch_size = size(X, 1)
    output_dim = size(model.output_w, 1)
    outputs = Matrix{Float64}(undef, batch_size, output_dim)
    traces = InferenceTrace[]

    for i in 1:batch_size
        trace = forward(model, @view X[i, :]; verbose=verbose)
        outputs[i, :] .= trace.final_output
        push!(traces, trace)
    end

    return outputs, traces
end

struct InferenceStepTrace
    steps::Vector{InferenceStep}
    final_output::Vector{Float64}
    total_steps::Int
    phase_history::Vector{PhaseType}
    slice_history::Vector{Int}
end

function forward_sequence(model::FourDModel, X_seq::AbstractArray{Float64, 3};
                          verbose::Bool=false)
    batch_size = size(X_seq, 1)
    seq_len = size(X_seq, 2)

    outputs = Matrix{Float64}(undef, batch_size, size(model.output_w, 1))
    all_traces = Vector{Vector{InferenceTrace}}(undef, batch_size)

    for b in 1:batch_size
        seq_traces = InferenceTrace[]
        accumulated_state = zeros(model.combined_output_dim)

        for t in 1:seq_len
            x_t = @view X_seq[b, t, :]
            mem_state = read_memory(model.memory)
            phase = RETRIEVE

            step_outputs = Vector{Float64}[]
            chosen_slices = Int[]
            chosen_phases = PhaseType[]

            for step in 1:model.max_steps
                phase_embed = encode_phase(model.phase_manager, phase)
                router_input = vcat(x_t, mem_state, phase_embed)
                route = router_forward(model.router, router_input)

                routed_slice_idx = route.chosen_slice
                chosen_phase, _ = decode_phase(model.phase_manager, route.phase_logits)

                compress_proj = model.slice_proj_ws[COMPRESS_IDX] * x_t .+ model.slice_proj_bs[COMPRESS_IDX]
                compress_out = slice_forward(model.slices[COMPRESS_IDX], compress_proj)
                compress_cond = film_forward(model.film_layers[COMPRESS_IDX], compress_out, phase_embed)

                routed_proj = model.slice_proj_ws[routed_slice_idx] * x_t .+ model.slice_proj_bs[routed_slice_idx]
                routed_out = slice_forward(model.slices[routed_slice_idx], routed_proj)
                routed_cond = film_forward(model.film_layers[routed_slice_idx], routed_out, phase_embed)

                combined = vcat(compress_cond, routed_cond)
                accumulated_state .+= combined

                push!(step_outputs, copy(combined))
                push!(chosen_slices, routed_slice_idx)
                push!(chosen_phases, chosen_phase)

                update_phase!(model.phase_manager, chosen_phase)
                phase = chosen_phase

                surprise_write!(model.memory, compress_cond)
                mem_state = read_memory(model.memory)

                if route.confidence > 0.8
                    break
                end
            end

            steps = [InferenceStep(chosen_slices[i], model.slices[chosen_slices[i]].name,
                                   chosen_phases[i], 0.5, 0.0, step_outputs[i])
                     for i in 1:length(step_outputs)]

            push!(seq_traces, InferenceTrace(
                steps, step_outputs[end], length(steps),
                chosen_phases, chosen_slices
            ))
        end

        accumulated_state ./= seq_len
        final_output = model.output_w * accumulated_state .+ model.output_b
        outputs[b, :] .= final_output
        all_traces[b] = seq_traces
    end

    return outputs, all_traces
end
