export PhaseManager, PhaseType, encode_phase, decode_phase, update_phase!
export RETRIEVE, REASON, PLAN, COMPRESS

using Random

@enum PhaseType begin
    RETRIEVE = 1
    REASON = 2
    PLAN = 3
    COMPRESS = 4
end

const N_PHASES = 4
const PHASE_EMBED_DIM = 8

mutable struct PhaseManager
    embedding::Matrix{Float64}
    transition_logits::Matrix{Float64}
    current::PhaseType
    phase_history::Vector{PhaseType}
    rng::MersenneTwister
end

function PhaseManager(; seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()

    embedding = randn(rng, N_PHASES, PHASE_EMBED_DIM) .* 0.1
    embedding[1, :] .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    embedding[2, :] .= [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    embedding[3, :] .= [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    embedding[4, :] .= [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    transition_logits = zeros(N_PHASES, N_PHASES)

    return PhaseManager(
        embedding, transition_logits,
        RETRIEVE,
        PhaseType[],
        rng
    )
end

function encode_phase(pm::PhaseManager, phase::PhaseType)
    idx = Int(phase)
    return @view pm.embedding[idx, :]
end

function decode_phase(pm::PhaseManager, logits::AbstractVector{Float64})
    probs = softmax(logits)
    idx = argmax(probs)
    return PhaseType(idx), probs
end

function update_phase!(pm::PhaseManager, new_phase::PhaseType)
    push!(pm.phase_history, new_phase)
    pm.current = new_phase
    return new_phase
end

softmax(x::AbstractVector{Float64}) = begin
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    exp_x ./ sum(exp_x)
end

function get_phase_transition_prior(pm::PhaseManager, from::PhaseType)
    idx = Int(from)
    return @view pm.transition_logits[idx, :]
end
