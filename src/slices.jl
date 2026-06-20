export Slice, create_slices!, slice_forward, N_SLICES

using Random

const N_SLICES = 4
const N_ROUTED_SLICES = 3

mutable struct Slice
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    name::String
    param_count::Int
end

function Slice(input_dim::Int, hidden_dim::Int, output_dim::Int,
               n_layers::Int, name::String; seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()

    weights = Matrix{Float64}[]
    biases = Vector{Float64}[]
    param_count = 0

    dims = vcat(input_dim, fill(hidden_dim, n_layers - 1), output_dim)

    for i in 1:n_layers
        fan_in = dims[i]
        fan_out = dims[i + 1]

        W = randn(rng, fan_out, fan_in) .* sqrt(2.0 / fan_in)
        b = zeros(fan_out)

        push!(weights, W)
        push!(biases, b)
        param_count += fan_out * fan_in + fan_out
    end

    return Slice(weights, biases, name, param_count)
end

function _relu!(x::AbstractVector{Float64})
    @simd for i in eachindex(x)
        @inbounds x[i] = max(0.0, x[i])
    end
    return x
end

function slice_forward(slice::Slice, x::AbstractVector{Float64})
    h = copy(x)

    for i in 1:length(slice.weights)
        W = slice.weights[i]
        b = slice.biases[i]

        new_h = W * h .+ b

        if i < length(slice.weights)
            _relu!(new_h)
        end

        h = new_h
    end

    return h
end

function slice_forward_batch(slice::Slice, X::AbstractMatrix{Float64})
    batch_size = size(X, 1)
    output_dim = size(slice.weights[end], 1)
    result = Matrix{Float64}(undef, batch_size, output_dim)

    @simd for i in 1:batch_size
        result[i, :] .= slice_forward(slice, @view X[i, :])
    end

    return result
end

function create_slices!(input_dim::Int, hidden_dim::Int=16, output_dim::Int=8;
                        seed::Union{Int,Nothing}=nothing)
    rng_base = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()

    slices = Slice[]

    push!(slices, Slice(input_dim, hidden_dim, output_dim, 2, "RETRIEVE";
                        seed=rand(rng_base, 1:10000)))

    push!(slices, Slice(input_dim, hidden_dim, output_dim, 3, "REASON";
                        seed=rand(rng_base, 1:10000)))

    push!(slices, Slice(input_dim, hidden_dim * 2, output_dim, 2, "PLAN";
                        seed=rand(rng_base, 1:10000)))

    push!(slices, Slice(input_dim, div(hidden_dim, 2), output_dim, 3, "COMPRESS";
                        seed=rand(rng_base, 1:10000)))

    return slices
end
