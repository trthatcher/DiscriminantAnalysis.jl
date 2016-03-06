#==========================================================================
  Regularized Quadratic Discriminant Analysis Solver
==========================================================================#

immutable ModelQDA{T<:BlasReal}
    W_k::Vector{Matrix{T}}  # Vector of class whitening matrices
    M::Matrix{T}             # Matrix of class means (one per row)
    priors::Vector{T}        # Vector of class priors
    gamma::Nullable{T}
    lambda::Nullable{T}
end

# λ-regularized QDA - require full covariance matrices
function class_whiteners!{T<:BlasReal,U<:Integer}(
        H::Matrix{T},
        y::RefVector{U}, 
        γ::Nullable{T},
        λ::T
    )
    f_k = one(T) ./ (class_counts(y) .- one(U))
    Σ_k = Matrix{T}[gramian(H[y .== i,:], f_k[i]) for i = 1:y.k]
    Σ   = gramian(H, one(T)/(size(H,1)-1))
    for S in Σ_k 
        regularize!(S, λ, Σ)
        whiten_cov!(S, γ)
    end
    Σ_k
end

# No λ-regularization - only need data matrices
function class_whiteners!{T<:BlasReal,U<:Integer}(H::Matrix{T}, y::RefVector{U}, γ::Nullable{T})
    Matrix{T}[whiten_data!(H[y .== i,:], γ) for i = 1:y.k]
end

function qda!{T<:BlasReal,U<:Integer}(
        X::Matrix{T}, 
        M::Matrix{T}, 
        y::RefVector{U}, 
        γ::Nullable{T},
        λ::Nullable{T}
    )
    H = center_classes!(X, M, y)
    isnull(λ) ? class_whiteners!(H, y, γ) : class_whiteners!(H, y, γ, get(λ))
end

doc"`qda(X, y; M, lambda, gamma, priors)` Fits a regularized quadratic discriminant model to the 
data in `X` based on class identifier `y`."
function qda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::AbstractVector{U};
        M::Matrix{T} = class_means(X,RefVector(y)),
        gamma::Union{T,Nullable{T}}  = Nullable{T}(),
        lambda::Union{T,Nullable{T}} = Nullable{T}(),
        priors::Vector{T} = ones(T,maximum(y))/maximum(y)
    )
    all(priors .> 0) || error("Argument priors must have positive values only")
    isapprox(sum(priors), one(T)) || error("Argument priors must sum to 1")
    γ = isa(gamma,  Nullable) ? deepcopy(gamma)  : Nullable(gamma)
    λ = isa(lambda, Nullable) ? deepcopy(lambda) : Nullable(lambda)
    W_k = qda!(copy(X), M, isa(y,RefVector) ? y : RefVector(y), γ, λ)
    ModelQDA{T}(W_k, M, priors, γ, λ)
end

function discriminants_qda{T<:BlasReal}(
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
    hᵀh = Array(T, n)
    for j = 1:k
        translate!(copy!(H, Z), -vec(M[j,:]))
        dotrows!(BLAS.gemm!('N', 'N', one(T), H, W_k[j], zero(T), Q), hᵀh)
        for i = 1:n
            δ[i, j] = -hᵀh[i]/2 + log(priors[j])
        end
    end
    δ
end

doc"`discriminants(Model, Z)` Uses `Model` on input `Z` to product the class discriminants."
function discriminants{T<:BlasReal}(mod::ModelQDA{T}, Z::Matrix{T})
    discriminants_qda(mod.W_k, mod.M, mod.priors, Z)
end

doc"`classify(Model, Z)` Uses `Model` on input `Z`."
function classify{T<:BlasReal}(mod::ModelQDA{T}, Z::Matrix{T})
    mapslices(indmax, discriminants(mod, Z), 2)
end
