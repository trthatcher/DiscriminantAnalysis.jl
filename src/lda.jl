#==========================================================================
  Regularized Linear Discriminant Analysis Solver
==========================================================================#

immutable ModelLDA{T<:BlasReal}
    W::Matrix{T}       # Whitening matrix
    M::Matrix{T}       # Matrix of class means (one per row)
    priors::Vector{T}  # Vector of class priors
end

# Fit regularized quadratic discriminant model. Returns whitening matrix for class variance.
#   X in uncentered data matrix
#   M is matrix of class means (one per row)
#   y is one-based vector of class IDs
#   λ is regularization parameter in [0,1]. λ = 0 is no regularization. See documentation.
function lda!{T<:BlasReal,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::Vector{U}, γ::T)
    k    = maximum(y)
    n_k  = class_counts(y, k)
    n, p = size(X)
    H    = center_classes!(X, M, y)
    w_σ  = 1 ./ vec(sqrt(var(H, 1)))  # Variance normalizing factor for columns of H
    scale!(H, w_σ)
    tol = eps(T) * prod(size(H)) * maximum(H)
    _U, D, Vᵀ = LAPACK.gesdd!('A', H)  # Recall: Sw = H'H/(n-1) = VD²Vᵀ
    @inbounds for i = 1:p
        D[i] /= sqrt(n - one(T))  # Note: we did not divide by n-1 above, so we do it now.
    end
    if γ > 0
        Λ = D.^2         # Since Sw = VD²Vᵀ, we have:
        λ_avg = mean(Λ)  #   (1-γ)Sw + γ(λ_avg)I = V((1-γ)D² + γ(λ_avg)I)Vᵀ
        for i = 1:p
            D[i] = sqrt((1-γ)*Λ[i] + γ*λ_avg)
        end
    end
    all(D .>= tol) || error("Rank deficiency (collinearity) detected.")
    scale!(one(T) ./ D, Vᵀ)
    scale!(w_σ, transpose(Vᵀ))
end

doc"`lda(X, y; M, gamma, priors)` Fits a regularized linear discriminant model to the data in `X` 
based on class identifier `y`."
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
    W * transpose(Vᵀ[1:k-1,:])
end

doc"`cda(X, y; M, gamma, priors)` Fits a regularized canonical discriminant model to the data in
`X` based on class identifier `y`."
function cda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::Vector{U};
        M::Matrix{T} = class_means(X,y),
        priors::Vector{T} = ones(T, maximum(y))/maximum(y),
        gamma::T = zero(T)
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
