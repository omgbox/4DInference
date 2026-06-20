export FilmLayer, film_forward, film_init!

using Random

struct FilmLayer
    gamma_w::Matrix{Float64}
    gamma_b::Vector{Float64}
    beta_w::Matrix{Float64}
    beta_b::Vector{Float64}
end

function FilmLayer(phase_dim::Int, feature_dim::Int; seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister()

    gamma_w = randn(rng, feature_dim, phase_dim) .* 0.01
    gamma_b = ones(feature_dim) .* 0.1

    beta_w = randn(rng, feature_dim, phase_dim) .* 0.01
    beta_b = zeros(feature_dim)

    return FilmLayer(gamma_w, gamma_b, beta_w, beta_b)
end

function film_forward(layer::FilmLayer, features::AbstractVector{Float64},
                      phase_embed::AbstractVector{Float64})
    gamma = layer.gamma_w * phase_embed .+ layer.gamma_b
    beta = layer.beta_w * phase_embed .+ layer.beta_b
    return gamma .* features .+ beta
end

function film_forward_batch(layer::FilmLayer, features::AbstractMatrix{Float64},
                            phase_embed::AbstractVector{Float64})
    batch_size = size(features, 1)
    feature_dim = size(features, 2)

    gamma = layer.gamma_w * phase_embed .+ layer.gamma_b
    beta = layer.beta_w * phase_embed .+ layer.beta_b

    result = similar(features)
    @simd for i in 1:batch_size
        @simd for j in 1:feature_dim
            @inbounds result[i, j] = gamma[j] * features[i, j] + beta[j]
        end
    end

    return result
end

function film_init!(layer::FilmLayer; gamma_init::Float64=1.0, beta_init::Float64=0.0)
    fill!(layer.gamma_b, gamma_init)
    fill!(layer.beta_b, beta_init)
    fill!(layer.gamma_w, 0.0)
    fill!(layer.beta_w, 0.0)
    return nothing
end
