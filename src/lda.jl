#==========================================================================
  Regularized Linear Discriminant Analysis Solver
==========================================================================#

immutable ModelLDA{T<:BlasReal}
    order::Union{Type{Val{:row}},Type{Val{:col}}}
    is_cda::Bool
    W::Matrix{T}       # Whitening matrix
    M::Matrix{T}       # Matrix of class means (one per row)
    priors::Vector{T}  # Vector of class priors
    gamma::Nullable{T}
end

function show(io::IO, model::ModelLDA)
    println(io, (model.is_cda ? "Canonical" : "Linear") * " Discriminant Model")
    println(io, "\nRegularization Parameters:")
    println(io, "γ = ", isnull(model.gamma) ? "n/a" : string(get(model.gamma)))
    println(io, "\nPrior Probabilities:")
    for i in eachindex(model.priors)
        println(io, "Class ", i, ": ", model.priors[i])
    end
    println(io, "\nGroup Means (one per row):")
    println(io, model.M)
end

#   X in uncentered data matrix
#   M is matrix of class means (one per row)
#   y is one-based vector of class IDs
#   λ is nullable regularization parameter in [0,1]
function lda!{T<:BlasReal,U}(
         ::Type{Val{:row}},
        X::Matrix{T},
        M::Matrix{T},
        y::RefVector{U},
        γ::Nullable{T}
    )
    H = centerclasses!(Val{:row}, X, M, y)
    isnull(γ) ? whitendata_qr!(H) : transpose(whitendata_svd!(H, get(γ)))
end

function lda!{T<:BlasReal,U}(
         ::Type{Val{:col}},
        X::Matrix{T},
        M::Matrix{T},
        y::RefVector{U},
        γ::Nullable{T}
    )
    H = centerclasses!(Val{:col}, X, M, y)
    isnull(γ) ? transpose!(whitendata_qr!(transpose(H))) : whitendata_svd!(transpose(H), get(γ))
end

doc"`lda(X, y; M, gamma, priors)` Fits a regularized linear discriminant model to the data in `X` 
based on class identifier `y`."
function lda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::AbstractVector{U};
        order::Union{Type{Val{:row}},Type{Val{:col}}} = Val{:row},
        M::Matrix{T} = classmeans(order, X, RefVector(y)),
        gamma::Union{T,Nullable{T}} = Nullable{T}(),
        priors::Vector{T} = ones(T,maximum(y))/maximum(y)
    )
    all(priors .> 0) || error("Argument priors must have positive values only")
    isapprox(sum(priors), one(T)) || error("Argument priors must sum to 1")
    γ = isa(gamma, Nullable) ? gamma : Nullable(gamma)
    W = lda!(order, copy(X), M, isa(y,RefVector) ? y : RefVector(y), γ)
    ModelLDA{T}(order, false, W, M, priors, γ)
end

function cda!{T<:BlasReal,U}(
         ::Type{Val{:row}},
        X::Matrix{T}, 
        M::Matrix{T}, 
        y::RefVector{U}, 
        γ::Nullable{T},
        priors::Vector{T}
    )
    k = convert(Int64, y.k)
    k == length(priors) || error("Argument priors length does not match class count")
    W_lda = lda!(Val{:row}, X, M, y, γ)
    Hm = broadcast!(-, M, M, priors'M)
    UDVᵀ = svdfact!(Hm * W_lda)
    V = transpose((UDVᵀ[:Vt])[1:min(k-1,size(X,2)),:]) # sub(UDVᵀ[:Vt], 1:min(y.k-1, size(X,2)), :)
    W = W_lda * V
end

function cda!{T<:BlasReal,U}(
         ::Type{Val{:col}},
        X::Matrix{T}, 
        M::Matrix{T}, 
        y::RefVector{U}, 
        γ::Nullable{T},
        priors::Vector{T}
    )
    k = convert(Int64, y.k)
    k == length(priors) || error("Argument priors length does not match class count")
    W_lda = lda!(Val{:col}, X, M, y, γ)
    Hm = broadcast!(-, M, M, M'priors)
    UDVᵀ = svdfact!(transpose(W_lda * Hm))
    Vᵀ = (UDVᵀ[:Vt])[1:min(k-1,size(X,2)),:] # sub(UDVᵀ[:Vt], 1:min(y.k-1, size(X,2)), :)
    W = Vᵀ * W_lda
end

doc"`cda(X, y; M, gamma, priors)` Fits a regularized canonical discriminant model to the data in
`X` based on class identifier `y`."
function cda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::AbstractVector{U};
        M::Matrix{T} = class_means(X,RefVector(y)),
        priors::Vector{T} = ones(T, maximum(y))/maximum(y),
        gamma::Union{T,Nullable{T}} = Nullable{T}()
    )
    all(priors .> 0) || error("Argument priors must have positive values only")
    isapprox(sum(priors), one(T)) || error("Argument priors must sum to 1")
    γ = isa(gamma, Nullable) ? gamma : Nullable(gamma)
    W = cda!(copy(X), copy(M), RefVector(y), γ, priors)
    ModelLDA{T}(true, W, M, priors, γ)
end

function discriminants_lda{T<:BlasReal}(
        W::Matrix{T},
        M::Matrix{T},
        priors::Vector{T},
        Z::Matrix{T}
    )
    n, p = size(Z)
    p == size(W, 1) || throw(DimensionMismatch("oops"))
    d = size(W, 2)
    k = length(priors)
    δ   = Array(T, n, k) # discriminant function values
    H   = Array(T, n, p) # temporary array to prevent re-allocation k times
    Q   = Array(T, n, d) # Q := H*W
    hᵀh = Array(T, n)    # diag(H'H)
    for j = 1:k
        translate!(copy!(H, Z), -vec(M[j,:]))
        dotrows!(BLAS.gemm!('N', 'N', one(T), H, W, zero(T), Q), hᵀh)
        for i = 1:n
            δ[i, j] = -hᵀh[i]/2 + log(priors[j])
        end
    end
    δ
end

function discriminants{T<:BlasReal}(mod::ModelLDA{T}, Z::Matrix{T})
    discriminants_lda(mod.W, mod.M, mod.priors, Z)
end

function classify{T<:BlasReal}(mod::ModelLDA{T}, Z::Matrix{T})
    mapslices(indmax, discriminants(mod, Z), 2)
end
