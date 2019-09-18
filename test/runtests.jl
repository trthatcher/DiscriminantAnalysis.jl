using Test, LinearAlgebra, Statistics, DiscriminantAnalysis

const DA = DiscriminantAnalysis


function random_data(T::Type{<:AbstractFloat}, nₘ::Vector{Int}, p::Int)
    n = sum(nₘ)
    m = length(nₘ)

    X = zeros(T, p, n)
    M = zeros(T, p, m)

    ix = sortperm(rand(n))

    y = [k for (k, nₖ) in enumerate(nₘ) for i = 1:nₖ][ix]

    for (k, nₖ) in enumerate(nₘ)
        Xₖ = randn(T, p, nₖ)
        Xₖ .-= mean(Xₖ, dims=2)

        μ = T[rand() < 0.5 ? -2k : 2k for i = 1:p]

        Xₖ .+= μ

        X[:, y .== k] = Xₖ
        M[:, k] = μ
    end

    return (X, y, M)
end

function random_cov(T::Type{<:AbstractFloat}, p::Integer)
    Q = qr(randn(T, p, p)).Q
    D = Diagonal(2p*rand(T, p))

    Σ = Symmetric(Q*D*transpose(Q))

    W_svd = Q*inv(√(D))
    W_chol = inv(cholesky(Σ).U)

    return (Σ, W_svd, W_chol)
end

function perturb(M::Matrix{T}, tol::Real=convert(T,0.05)) where {T <: Real}
    ϵ = convert(T, tol)
    jitter = range(one(T)-ϵ, stop=one(T)+ϵ, length=10000)
    return broadcast(x -> x*rand(jitter), M)
end

@testset "DiscriminantAnalysis.jl" begin
    include("test_common.jl")
    include("test_lda.jl")
end
