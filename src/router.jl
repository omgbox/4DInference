export FourDRouter, create_router!, router_forward, RouterOutput

using Random

# Uses N_ROUTED_SLICES from slices.jl (included before router.jl)

struct RouterOutput
    slice_logits::Vector{Float64}
    slice_probs::Vector{Float64}
    chosen_slice::Int
    phase_logits::Vector{Float64}
    phase_probs::Vector{Float64}
    confidence::Float64
    hidden::Vector{Float64}
end

mutable struct FourDRouter
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}

    slice_head_w::Matrix{Float64}
    slice_head_b::Vector{Float64}
    phase_head_w::Matrix{Float64}
    phase_head_b::Vector{Float64}
    confidence_head_w::Vector{Float64}
    confidence_head_b::Float64

    hidden_dim::Int
    rng::MersenneTwister
end

function create_router!(input_dim::Int, hidden_dim::Int=16;
                        seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()

    W1 = randn(rng, hidden_dim, input_dim) .* sqrt(2.0 / input_dim)
    b1 = zeros(hidden_dim)

    W2 = randn(rng, hidden_dim, hidden_dim) .* sqrt(2.0 / hidden_dim)
    b2 = zeros(hidden_dim)

    slice_head_w = randn(rng, N_ROUTED_SLICES, hidden_dim) .* 0.01
    slice_head_b = zeros(N_ROUTED_SLICES)

    phase_head_w = randn(rng, 4, hidden_dim) .* 0.01
    phase_head_b = zeros(4)

    confidence_head_w = randn(rng, hidden_dim) .* 0.01
    confidence_head_b = 0.0

    return FourDRouter(
        W1, b1, W2, b2,
        slice_head_w, slice_head_b,
        phase_head_w, phase_head_b,
        confidence_head_w, confidence_head_b,
        hidden_dim, rng
    )
end

function router_forward(router::FourDRouter, x::AbstractVector{Float64})
    h1 = router.W1 * x .+ router.b1
    @simd for i in eachindex(h1)
        @inbounds h1[i] = max(0.0, h1[i])
    end

    h2 = router.W2 * h1 .+ router.b2
    @simd for i in eachindex(h2)
        @inbounds h2[i] = max(0.0, h2[i])
    end

    slice_logits = router.slice_head_w * h2 .+ router.slice_head_b
    slice_probs = _softmax(slice_logits)
    chosen_slice = argmax(slice_probs)

    phase_logits = router.phase_head_w * h2 .+ router.phase_head_b
    phase_probs = _softmax(phase_logits)

    conf_raw = router.confidence_head_w' * h2 + router.confidence_head_b
    confidence = 1.0 / (1.0 + exp(-conf_raw))

    return RouterOutput(
        slice_logits, slice_probs, chosen_slice,
        phase_logits, phase_probs,
        confidence,
        h2
    )
end

function _softmax(x::AbstractVector{Float64})
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    exp_x ./ sum(exp_x)
end
