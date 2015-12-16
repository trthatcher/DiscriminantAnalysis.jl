#==========================================================================
  Regularized Linear Discriminant Analysis Solver
==========================================================================#

immutable ModelLDA{T<:BlasReal}
    W::Matrix{T}       # Whitening matrix
    M::Matrix{T}       # Matrix of class means (one per row)
    priors::Vector{T}  # Vector of class priors
end

function lda!{T<:BlasReal,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::Vector{U}, γ::T)
    k = maximum(y)
    n_k = class_counts(y, k)
    n, p = size(X)
    H = center_classes!(X, M, y)
    w_σ = one(T) ./ vec(sqrt(var(H, 1)))  # scaling constant vector
    scale!(H, w_σ)
    _U, D, Vᵀ = LAPACK.gesdd!('A', H)  # Sw = H'H/(n-1)
    if γ > 0
        μ_λ = mean(D.^2)
        for i = 1:p
            D[i] = (1-γ)*D[i] + γ*μ_λ
        end
    end
    for i = 1:p
        D[i] != 0 || error("Rank deficiency (collinearity) detected.")
        D[i] = one(T)/D[i]
    end
    scale!(D, Vᵀ)
    scale!(w_σ, transpose(Vᵀ))
end

doc"""
`lda(X, y; M, gamma, priors)`
Fits a regularized linear discriminant model to the data in `X` based on class identifier `y`. 
"""
function lda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::Vector{U};
        M::Matrix{T} = class_means(X,y),
        gamma::T = zero(T),
        priors::Vector{T} = ones(T,maximum(y))/maximum(y)
    )
    W = lda!(copy(X), M, y, gamma)
    ModelLDA{T}(W, M, priors)
end

function cda!{T<:BlasReal,U<:Integer}(
        X::Matrix{T}, 
        M::Matrix{T}, 
        y::Vector{U}, 
        priors::Vector{T}, 
        γ::T
    )
    k = length(priors)
    W = lda!(X, M, y, γ)
    μ = vec(priors'M)
    H_mw = translate!(M, -μ) * W  # H_mw := (M .- μ') * W
    _U, D, Vᵀ = LAPACK.gesdd!('A', H_mw)
    W * transpose(Vᵀ[1:k-1,:])  #[end-k+1:end,:])
end

doc"""
`cda(X, y; M, gamma, priors)`
Fits a regularized canonical discriminant model to the data in `X` based on class identifier `y`. 
"""
function cda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::Vector{U};
        M::Matrix{T} = class_means(X,y),
        gamma::T = zero(T),
        priors::Vector{T} = T[1/maximum(y) for i = 1:maximum(y)]
    )
    W = cda!(copy(X), copy(M), y, priors, gamma)
    ModelLDA{T}(W, M, priors)
end

function classify_lda{T<:BlasReal}(
        W::Matrix{T},
        M::Matrix{T},
        priors::Vector{T},
        Z::Matrix{T}
    )
    n, p = size(Z)
    p == size(W, 1) || throw(DimensionMismatch("oops"))
    d = size(W, 2)
    k = length(priors)
    δ = Array(T, n, k)  # discriminant function values
    H = Array(T, n, p)  # temporary array to prevent re-allocation k times
    Q = Array(T, n, d)  # Q := H*W
    for j = 1:k
        translate!(copy!(H, Z), -vec(M[j,:]))
        s = dot_rows(BLAS.gemm!('N', 'N', one(T), H, W, zero(T), Q))
        for i = 1:n
            δ[i, j] = -s[i]/2 + log(priors[j])
        end
    end
    mapslices(indmax, δ, 2)
end

classify{T<:BlasReal}(mod::ModelLDA{T}, Z::Matrix{T}) = classify_lda(mod.W, mod.M, mod.priors, Z)
