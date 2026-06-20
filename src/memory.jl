export SORNMemory, create_sorn_memory!, surprise_write!, read_memory

using Random
using SparseArrays

mutable struct SORNMemory
    W::SparseMatrixCSC{Float64,Int}
    state::Vector{Float64}
    trace::Vector{Float64}
    n_neurons::Int
    input_dim::Int
    threshold::Float64
    trace_decay::Float64
    w_max::Float64
    a_plus::Float64
    a_minus::Float64
    rng::MersenneTwister
end

function create_sorn_memory!(input_dim::Int; n_neurons::Int=10,
                             connectivity::Float64=0.15,
                             threshold::Float64=0.3,
                             trace_decay::Float64=0.9,
                             w_max::Float64=1.0,
                             seed::Union{Int,Nothing}=nothing)

    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()

    n = n_neurons
    W = spzeros(n, n)
    for j in 1:n, i in 1:n
        i != j && rand(rng) < connectivity && (W[i, j] = 0.1 + 0.9 * rand(rng))
    end

    W_in = spzeros(n, input_dim)
    for j in 1:input_dim, i in 1:n
        rand(rng) < connectivity && (W_in[i, j] = 0.5 + 0.5 * rand(rng))
    end

    state = zeros(n)
    trace = zeros(n)

    return SORNMemory(
        W, state, trace,
        n, input_dim,
        threshold, trace_decay, w_max,
        0.0005, 0.0007,
        rng
    )
end

function surprise_write!(mem::SORNMemory, input::AbstractVector{Float64})
    @simd for i in 1:min(length(input), mem.n_neurons)
        @inbounds mem.state[i] = 0.9 * mem.state[i] + 0.1 * input[i]
    end

    input_truncated = @view input[1:min(length(input), mem.n_neurons)]
    state_truncated = @view mem.state[1:min(length(input), mem.n_neurons)]
    surprise = zero(Float64)
    @simd for i in eachindex(input_truncated)
        @inbounds surprise += (input_truncated[i] - state_truncated[i])^2
    end
    surprise = sqrt(surprise / length(input_truncated))

    if surprise > mem.threshold
        rv = rowvals(mem.W)
        nz = nonzeros(mem.W)

        @inbounds for j in 1:mem.n_neurons
            if abs(mem.state[j]) > 0.3
                pt_j = mem.trace[j]
                for idx in nzrange(mem.W, j)
                    i = rv[idx]
                    if abs(mem.state[i]) > 0.3
                        nz[idx] += mem.a_plus * (1.0 - nz[idx] / mem.w_max) * pt_j
                        nz[idx] -= mem.a_minus * pt_j
                        nz[idx] = clamp(nz[idx], 0.0, mem.w_max)
                    end
                end
            end
        end
    end

    @simd for i in 1:mem.n_neurons
        @inbounds mem.trace[i] = mem.trace_decay * mem.trace[i] + abs(mem.state[i])
    end

    return mem.state
end

function read_memory(mem::SORNMemory)
    return copy(mem.state)
end
