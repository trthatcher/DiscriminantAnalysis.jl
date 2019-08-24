using Test, LinearAlgebra, Statistics, DiscriminantAnalysis

const DA = DiscriminantAnalysis


function generate_data(T::Type{<:AbstractFloat}, n::Int, p::Int)
    ix = sortperm(rand(2n))

    y = repeat(1:2, inner = n)[ix]

    X1 = rand(T, n, p)
    X1 .-= mean(X1, dims = 1)
    X1 .+= 1
    X2 = rand(T, n, p)
    X2 .-= mean(X2, dims = 1)
    X2 .-= 1
    
    X = [X1; X2][ix, :]

    M = [mean(X1, dims = 1); mean(X2, dims = 1)]

    (X, y, M)
end

function random_cov(T::Type{<:AbstractFloat}, p::Integer)
    Q = qr(randn(T, p, p)).Q
    D = Diagonal(2p*rand(T, p))

    Σ = Symmetric(Q*D*transpose(Q))

    W_svd = Q*inv(√(D))
    W_chol = inv(cholesky(Σ).U)

    return (Σ, W_svd, W_chol)
end

@testset "DiscriminantAnalysis.jl" begin
    @info "Testing common.jl"
    include("test_common.jl")

    #@info "Testing lda.jl"
    #include("test_lda.jl")
end
