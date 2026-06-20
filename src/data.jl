export generate_dataset, DataLoader, next_batch!, generate_sequence_dataset

using Random

const LABEL_NAMES = ["deny", "review", "standard", "approve", "premium", "platinum"]

struct Sample
    features::Vector{Float64}
    label::Int
    difficulty::Int
end

function classify(age::Float64, income::Float64, credit::Float64)
    if age < 0.17
        return 1, 1
    end
    if income < 0.1
        return 1, 1
    end

    if age >= 0.17 && age < 0.33 && income >= 0.25
        if credit > 0.67
            return 4, 2
        else
            return 2, 2
        end
    end

    if age >= 0.5 && income >= 0.6
        if credit > 0.93
            return 6, 3
        elseif credit > 0.83
            return 5, 3
        else
            return 3, 3
        end
    end

    if age >= 0.33 && income >= 0.4
        if credit > 0.83
            return 5, 3
        elseif credit > 0.67
            return 3, 3
        else
            return 2, 3
        end
    end

    return 3, 2
end

function generate_dataset(n_samples::Int; seed::Union{Int,Nothing}=nothing,
                          difficulty::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister(42)

    X = Matrix{Float64}(undef, n_samples, 3)
    y = Vector{Int}(undef, n_samples)
    diff = Vector{Int}(undef, n_samples)

    for i in 1:n_samples
        age = rand(rng)
        income = rand(rng)
        credit = rand(rng)

        if difficulty !== nothing
            if difficulty == 1
                age = rand(rng) * 0.17
                income = rand(rng) * 0.1
            elseif difficulty == 2
                age = 0.17 + rand(rng) * 0.16
                income = 0.25 + rand(rng) * 0.15
                credit = rand(rng)
            elseif difficulty == 3
                age = 0.33 + rand(rng) * 0.67
                income = 0.4 + rand(rng) * 0.6
                credit = 0.67 + rand(rng) * 0.33
            end
        end

        X[i, :] .= [age, income, credit]
        label, d = classify(age, income, credit)
        y[i] = label
        diff[i] = d
    end

    return X, y, diff
end

function generate_sequence_dataset(n_samples::Int; seq_len::Int=5,
                                   seed::Union{Int,Nothing}=nothing)
    rng = seed !== nothing ? MersenneTwister(seed) : MersenneTwister(42)

    X = Array{Float64}(undef, n_samples, seq_len, 3)
    y = Vector{Int}(undef, n_samples)
    diff = Vector{Int}(undef, n_samples)

    for i in 1:n_samples
        base_age = rand(rng)
        base_income = rand(rng)
        base_credit = rand(rng)

        for t in 1:seq_len
            noise_age = randn(rng) * 0.05
            noise_income = randn(rng) * 0.05
            noise_credit = randn(rng) * 0.05

            age = clamp(base_age + noise_age * t, 0.0, 1.0)
            income = clamp(base_income + noise_income * t, 0.0, 1.0)
            credit = clamp(base_credit + noise_credit * t, 0.0, 1.0)

            X[i, t, :] .= [age, income, credit]
        end

        final_age = X[i, end, 1]
        final_income = X[i, end, 2]
        final_credit = X[i, end, 3]

        label, d = classify(final_age, final_income, final_credit)
        y[i] = label
        diff[i] = d
    end

    return X, y, diff
end

struct DataLoader
    X::Matrix{Float64}
    y::Vector{Int}
    difficulty::Vector{Int}
    batch_size::Int
    n_samples::Int
    indices::Vector{Int}
    pos::Base.RefValue{Int}

    function DataLoader(X::Matrix{Float64}, y::Vector{Int}, difficulty::Vector{Int};
                       batch_size::Int=32)
        n = size(X, 1)
        new(X, y, difficulty, batch_size, n, shuffle(1:n), Ref(1))
    end
end

function next_batch!(dl::DataLoader)
    if dl.pos[] + dl.batch_size - 1 > dl.n_samples
        dl.indices .= shuffle(1:dl.n_samples)
        dl.pos[] = 1
    end

    batch_range = dl.pos[]:(dl.pos[] + dl.batch_size - 1)
    idx = dl.indices[batch_range]
    dl.pos[] += dl.batch_size

    return dl.X[idx, :], dl.y[idx], dl.difficulty[idx]
end

function Base.length(dl::DataLoader)
    return cld(dl.n_samples, dl.batch_size)
end

function Base.length(dl::DataLoader)
    return cld(dl.n_samples, dl.batch_size)
end
