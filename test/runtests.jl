using Test, LinearAlgebra, Statistics, DiscriminantAnalysis

const DA = DiscriminantAnalysis

function random_centroids(T::Type{<:AbstractFloat}, m::Int, p::Int)
    M = zeros(T, p, m)
    for k = 1:m
        M[:, k] = T[rand() < 0.5 ? -2k : 2k for i = 1:p]
    end
    return M
end

function random_data(T::Type{<:AbstractFloat}, nₘ::Vector{Int}, p::Int)
    n = sum(nₘ)
    m = length(nₘ)

    X = zeros(T, p, n)
    M = zeros(T, p, m)

    ix = sortperm(rand(n))

    y = [k for (k, nₖ) in enumerate(nₘ) for i = 1:nₖ][ix]
    M = random_centroids(T, m, p)

    for (k, nₖ) in enumerate(nₘ)
        Xₖ = randn(T, p, nₖ)

        Xₖ .-= mean(Xₖ, dims=2)
        Xₖ .+= view(M, :, k)

        X[:, y .== k] = Xₖ
    end

    return (X, y, M)
end

function random_cov(T::Type{<:AbstractFloat}, p::Integer)
    Q = qr(randn(T, p, p)).Q
    D = Diagonal(2p*rand(T, p))

    Σ = Symmetric(transpose(Q)*D*Q)

    W_svd = inv(√(D))*Q
    W_chol = copy(transpose(inv(cholesky(Σ).U)))

    return (Σ, W_svd, W_chol)
end

function perturb(M::Matrix{T}, tol::Real=convert(T,0.05)) where {T <: Real}
    ϵ = convert(T, tol)
    jitter = range(one(T)-ϵ, stop=one(T)+ϵ, length=10000)
    return broadcast(x -> x*rand(jitter), M)
end

@testset "DiscriminantAnalysis.jl" begin
    include("test_common.jl")
    include("test_whiten.jl")
    include("test_discriminant.jl")
    include("test_linear.jl")
end
