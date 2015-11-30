#==========================================================================
  Regularized Quadratic Discriminant Analysis Solver
==========================================================================#

immutable ModelQDA{T<:BlasReal}
    W_k::Array{Matrix{T},1}  # Vector of class whitening matrices
    M::Matrix{T}             # Matrix of class means (one per row)
    priors::Vector{T}        # Vector of class priors
end

# Create an array of class scatter matrices
#   H is centered data matrix (with respect to class means)
#   y is one-based vector of class IDs
function class_covariances{T<:BlasReal,U<:Integer}(H::Matrix{T}, y::Vector{U}, 
                                                   n_k::Vector{Int64} = class_counts(y))
    k = length(n_k)
    p = size(H,2)
    Σ_k = Array(Array{T,2}, k)
    for i = 1:k  # Σ_k[i] = H_i'H_i/(n_i-1)
        Σ_k[i] = BLAS.syrk!('U', 'T', one(T)/(n_k[i]-1), H[y .== i,:], zero(T), Array(T,p,p))
        symml!(Σ_k[i])
    end
    Σ_k
end

# Use eigendecomposition to generate class whitening transform
#   Σ_k is array of references to each Σ_i covariance matrix
#   λ is regularization parameter in [0,1]
function class_whiteners!{T<:BlasReal}(Σ_k::Vector{Matrix{T}}, γ::T)
    for i = 1:length(Σ_k)
        tol = eps(T) * prod(size(Σ_k[i])) * maximum(Σ_k[i])
        Λ_i, V_i = LAPACK.syev!('V', 'U', Σ_k[i])  # Overwrite Σ_k with V such that VΛVᵀ = Σ_k
        if γ > 0
            μ_λ = mean(Λ_i)  # Shrink towards average eigenvalue
            translate!(scale!(Λ_i, 1-γ), γ*μ_λ)  # Σ = VΛVᵀ => (1-γ)Σ + γI = V((1-γ)Λ + γI)Vᵀ
        end
        all(Λ_i .>= tol) || error("Rank deficiency detected in class $i with tolerance $tol.")
        scale!(V_i, one(T) ./ sqrt(Λ_i))  # Scale V so it whitens H*V where H is centered X
    end
    Σ_k
end

# Fit regularized discriminant model
#   X in uncentered data matrix
#   M is matrix of class means (one per row)
#   y is one-based vector of class IDs
function qda!{T<:BlasReal,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::Vector{U}, λ::T, γ::T)
    k = maximum(y)
    n_k = class_counts(y, k)
    n, p = size(X)
    H = center_classes!(X, M, y)
    w_σ = one(T) ./ vec(sqrt(var(H, 1)))  # scaling constant vector
    scale!(H, w_σ)
    Σ_k = class_covariances(H, y, n_k)
    if λ > 0
        Σ = scale!(H'H, one(T)/(n-1))
        for i = 1:k 
            regularize!(Σ_k[i], λ, Σ)
        end
    end
    W_k = class_whiteners!(Σ_k, γ)
    for i = 1:k
        scale!(w_σ, W_k[i])  # scale rows of W_k
    end
    W_k
end

doc"""
`qda(X, y; M, lambda, gamma, priors)`
Fits a regularized quadratic discriminant model to the data in `X` based on class identifier `y`.
"""
function qda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::Vector{U};
        M::Matrix{T} = class_means(X,y),
        lambda::T = zero(T),
        gamma::T = zero(T),
        priors::Vector{T} = T[1/maximum(y) for i = 1:maximum(y)]
    )
    W_k = qda!(copy(X), M, y, lambda, gamma)
    ModelQDA{T}(W_k, M, priors)
end

function classify_qda{T<:BlasReal}(
        W_k::Vector{Matrix{T}},
        M::Matrix{T},
        priors::Vector{T},
        Z::Matrix{T}
    )
    n, p = size(Z)
    k = length(priors)
    size(M,2) == p || throw(DimensionMismatch("Z does not have the same number of columns as M."))
    size(M,1) == k || error("class mismatch")
    length(W_k) == k || error("class mismatch")
    δ = Array(T, n, k)  # discriminant function values
    H = Array(T, n, p)  # temporary array to prevent re-allocation k times
    Q = Array(T, n, p)  # Q := H*W_k
    for j = 1:k
        translate!(copy!(H, Z), -vec(M[j,:]))
        s = dot_rows(BLAS.gemm!('N', 'N', one(T), H, W_k[j], zero(T), Q))
        for i = 1:n
            δ[i, j] = -s[i]/2 + log(priors[j])
        end
    end
    mapslices(indmax, δ, 2)
end

doc"""
`classify(Model, Z)`
Uses model on input Z.
"""
classify{T<:BlasReal}(mod::ModelQDA{T}, Z::Matrix{T}) = classify_qda(mod.W_k, mod.M, mod.priors, Z)
