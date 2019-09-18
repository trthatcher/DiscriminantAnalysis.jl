# Common Functions

### Dimensionality Checks

function check_dims(X::AbstractMatrix; dims::Integer)
    dims ∈ (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))

    n = size(X, dims)
    p = size(X, dims == 1 ? 2 : 1)

    return (n, p)
end


### Data/Parameter Validation

function validate_priors(π::AbstractVector{T}) where T
    total = zero(T)

    for (k, πₖ) in enumerate(π)
        if !(zero(T) < πₖ < one(T))
            throw(DomainError(πₖ, "class prior probability at class index $(k) must be " *
                                  "in the open interval (0,1)"))
        end
        total += πₖ
    end
    
    if !isapprox(total, one(T))
        throw(ArgumentError("class priors vector π must sum to 1 (got $(total))"))
    end

    return length(π)
end

function validate_class_counts(nₘ::AbstractVector{T}) where {T <: Integer}
    for (k, nₖ) in enumerate(nₘ)
        nₖ > 1 || error("class count at class index $(k) must be ")
    end

    return length(nₘ)
end


### Class Calculations

"""
    class_counts!(nₘ, y)

Overwrite vector `nₘ` with counts of each index in `y`
"""
function class_counts!(nₘ::Vector{T}, y::Vector{<:Integer}) where {T<:Integer}
    m = length(nₘ)

    nₘ .= zero(T)

    for i = 1:length(y)
        yᵢ = y[i]
        1 ≤ yᵢ ≤ m || throw(BoundsError(nₘ, yᵢ))
        nₘ[yᵢ] += 1
    end

    return nₘ
end

"""
    _class_statistics!(M, nₘ, X, y)

Backend for `class_statistics!` - assumes `dims=2`.
"""
function _class_statistics!(M::AbstractMatrix, nₘ::Vector{<:Integer}, X::AbstractMatrix, 
                            y::Vector{<:Integer})
    n, p = check_dims(X, dims=2)
    m, p₂ = check_dims(M, dims=2)
    n₂ = length(y)
    m₂ = length(nₘ)

    p == p₂ || throw(DimensionMismatch("predictor count along dimension 1 of X must " *
                                       "match dimension 1 of M (got $(p) and $(p₂))"))
    n == n₂ || throw(DimensionMismatch("observation count along length of y must match " *
                                       "dimension 2 of X (got $(n) and $(n₂))"))
    m == m₂ || throw(DimensionMismatch("class count along length of nₘ must match " *
                                       "dimension 1 of M (got $(m₂) and $(m))"))

    T = eltype(nₘ)

    M .= zero(eltype(M))
    nₘ .= zero(T)  # track counts to ensure an observation for each class

    for i = 1:n
        yᵢ = y[i]
        1 ≤ yᵢ ≤ m || throw(BoundsError(M, (1, yᵢ)))
        nₘ[yᵢ] += one(T)
        @inbounds for j = 1:p
            M[j, yᵢ] += X[j, i]
        end
    end

    all(nₘ .≥ 1) || error("must have at least one observation per class")
    broadcast!(/, M, M, transpose(nₘ))

    return (M, nₘ)
end

"""
    class_statistics!(M, nₘ, X, y; dims=1)

Overwrites matrix `M` with class centroids from `X` based on class indexes from `y`. Use
`dims=1` for row-based observations and `dims=2` for column-based observations.
"""
function class_statistics!(M::AbstractMatrix, nₘ::Vector{<:Integer}, X::AbstractMatrix, 
                           y::Vector{<:Integer}; dims::Integer=1)
    if dims == 1
        n, p = check_dims(X, dims=2)
        m, p₂ = check_dims(M, dims=2)
        n₂ = length(y)
        m₂ = length(nₘ)
    
        p == p₂ || throw(DimensionMismatch("predictor count along dimension 2 of X must " *
                                           "match dimension 2 of M (got $(p) and $(p₂))"))
        n == n₂ || throw(DimensionMismatch("observation count along length of y must " *
                                           "match dimension 1 of X (got $(n) and $(n₂))"))
        m == m₂ || throw(DimensionMismatch("class count along length of nₘ must match " *
                                           "dimension 2 of M (got $(m₂) and $(m))"))

        _class_statistics!(transpose(M), nₘ, transpose(X), y)
        return (M, nₘ)
    elseif dims ==2
        return _class_statistics!(M, nₘ, X, y)
    else
        throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    end
end


### Model Parameters

mutable struct DiscriminantParameters{T}
    "Model fit indicator - `true` if model has been fit"
    fit::Bool
    "Dimension along which observations are stored (1 for rows, 2 for columns)"
    dims::Int
    "Matrix of class centroids (one per row or column - see `dims`)"
    M::Matrix{T}
    "Prior-weighted overall centroid"
    μ::Vector{T}
    "Vector of class prior probabilities"
    π::Vector{T}
    "Vector of Class counts"
    nₘ::Vector{Int}
    "Shrinkage parameter"
    γ::Union{Nothing,T}
    "Overall covariance matrix without `γ` regularization"
    Σ::Union{Nothing,T}
    "Determinant of the overall covariance matrix"
    detΣ::Union{Nothing,T}
    function DiscriminantParameters{T}() where T
        new{T}(false, 0, Matrix{T}(undef,0,0), Vector{T}(undef,0), Vector{T}(undef,0),
               Vector{Int}(undef,0), nothing, nothing, zero(T))
    end
end

function parameter_fit!(Θ::DiscriminantParameters{T},
                        y::Vector{<:Integer},
                        X::Matrix{T},
                        dims::Integer,
                        compute_covariance::Bool,
                        centroids::Union{Nothing,AbstractMatrix}, 
                        priors::Union{Nothing,AbstractVector},
                        gamma::Union{Nothing,Real}) where T
    n, p = check_dims(X, dims=dims)
    m = maximum(y)
    
    n₂ = length(y)
    n₂ == n || throw(DimensionMismatch("observation count along length of class index " *
                                       "vector y must match dimension $(dims) of data " *
                                       "matrix X (got $(n₂) and $(n))"))

    Θ.dims = dims
    is_row = dims == 1
    altdims = is_row ? 2 : 1

    if gamma !== nothing
        0 ≤ gamma ≤ 1 || throw(DomainError(gamma, "γ must be in the interval [0,1]"))
    end
    Θ.γ = gamma

    # Compute centroids and class counts from data if not specified
    if centroids === nothing
        Θ.M = is_row ? Matrix{T}(undef, m, p) : Matrix{T}(undef, p, m)
        Θ.nₘ = Vector{Int}(undef, m)

        class_statistics!(Θ.M, Θ.nₘ, X, y, dims=dims)
    else
        m₂, p₂ = check_dims(centroids, dims=dims)
        if m₂ != m
            throw(DimensionMismatch("class count along dimension $(dims) of centroid " * 
                                    "matrix M must match maximum class index found in " *
                                    "class index vector y (got $(m₂) and $(m))"))
        elseif p₂ != p
            throw(DimensionMismatch("predictor count along dimension $(altdims) of " *
                                    "centroid matrix M must match dimension $(dims) of " *
                                    "data matrix X (got $(p₂) and $(p))"))
        end

        Θ.M = copyto!(similar(centroids, T), centroids)
        Θ.nₘ = class_counts!(Vector{Int}(undef, m), y)
    end

    validate_class_counts(Θ.nₘ)

    # Compute priors from class frequencies in data if not specified
    if priors === nothing
        Θ.π = broadcast!(/, Vector{T}(undef, m), Θ.nₘ, n)
    else
        m₃ = length(priors)
        if m₃ != m
            throw(DimensionMismatch("class count along length of class prior probability " *
                                    "vector π must match maximum class index found in " *
                                    "class index vector y (got $(m₃) and $(m))"))
        end
        validate_priors(priors)

        Θ.π = copyto!(similar(priors, T), priors)
    end

    # Overall centroid is prior-weighted average of class centroids
    Θ.μ = is_row ? vec(transpose(Θ.π)*Θ.M) : Θ.M*Θ.π

    # Center the data matrix with respect to classes to compute whitening matrix
    X .-= is_row ? view(Θ.M, y, :) : view(Θ.M, :, y)

    return Θ
end


### Regularization

"""
    regularize!(Σ₁, Σ₂, λ)
"""
function regularize!(Σ₁::AbstractMatrix{T}, Σ₂::AbstractMatrix{T}, λ::T) where {T}
    (m = size(Σ₁, 1)) == size(Σ₂, 1) || throw(DimensionMismatch(""))
    (n = size(Σ₁, 2)) == size(Σ₂, 2) || throw(DimensionMismatch(""))

    m == n || throw(DimensionMismatch("matrices Σ₁ and Σ₂ must be square"))

    0 ≤ λ ≤ 1 || throw(DomainError(λ, "λ must be in the interval [0,1]"))

    for ij in eachindex(Σ₁)
        Σ₁[ij] = (1-λ)*Σ₁[ij] + λ*Σ₂[ij]
    end
    
    return Σ₁
end


"""
    regularize!(Σ, γ)

Shrink `Σ` matrix towards the average eigenvalue multiplied by the identity matrix.
"""
function regularize!(Σ::AbstractMatrix{T}, γ::T) where {T}
    (p = size(Σ, 1)) == size(Σ, 2) || throw(DimensionMismatch("Σ must be square"))

    0 ≤ γ ≤ 1 || throw(DomainError(γ, "γ must be in the interval [0,1]"))

    a =  γ*tr(Σ)/p  # Average eigenvalue scaled by γ
    
    broadcast!(*, Σ, Σ, 1 - γ)
    for i = 1:p
        Σ[i, i] += a
    end

    return Σ
end


### Data Whitening Functions

"""
    whiten_data!(X; dims, df)

Compute a whitening transform matrix for centered data matrix `X`. Use `dims=1` for 
row-based observations and `dims=2` for column-based observations. The `df` parameter 
specifies the effective degrees of freedom.
"""
function whiten_data!(X::Matrix{T}; dims::Integer, df::Integer=size(X,dims)-1) where T
    df > 0 || error("degrees of freedom must be greater than 0")

    n, p = check_dims(X, dims=dims)

    n > p || error("insufficient number of within-class observations to produce a full " *
                   "rank covariance matrix ($(n) observations, $(p) predictors)")

    if dims == 1
        # X = QR ⟹ S = XᵀX = RᵀR
        R = UpperTriangular(qr!(X, Val(false)).R)  
    else
        # Xᵀ = LQ ⟹ S = XXᵀ = LLᵀ = RᵀR
        R = UpperTriangular(transpose(lq!(X).L))  
    end

    broadcast!(/, R, R, √(df))

    detΣ = det(R)^2

    W = try
        inv(R)
    catch err
        if isa(err, LAPACKException) || isa(err, SingularException)
            if err.info ≥ 1
                error("rank deficiency detected (collinearity in predictors)")
            end
        end
        throw(err)
    end

    if dims == 1
        return (W, detΣ)
    else
        return (copy(transpose(W)), detΣ)
    end
end

@inline regularize(x, y, γ) = (1-γ)*x + γ*y

function whiten_data!(X::Matrix{T}, γ::T; dims::Integer, df::Integer=size(X,dims)-1) where T
    n, p = check_dims(X, dims=dims)
    
    n > p || error("insufficient number of within-class observations to produce a full " *
                   "rank covariance matrix ($(n) observations, $(p) predictors)")
    
    0 ≤ γ ≤ 1 || throw(DomainError(γ, "γ must be in the interval [0,1]"))

    tol = eps(one(T))*p*maximum(X)

    UDVᵀ = svd!(X, full=false)

    D = UDVᵀ.S

    if γ ≠ zero(T)
        # Regularize: Σ = VD²Vᵀ ⟹ Σ(γ) = V((1-γ)D² + (γ/p)trace(D²)I)Vᵀ
        broadcast!(σᵢ -> abs2(σᵢ)/df, D, D)  # Convert data singular values to Σ eigenvalues
        broadcast!(regularize, D, D, mean(D), γ)
        detΣ = prod(D)
        broadcast!(√, D, D)
    else
        detΣ = prod(σᵢ -> abs2(σᵢ)/df, D)
        broadcast!(/, D, D, √(df))
    end

    all(D .> tol) || error("rank deficiency (collinearity) detected with tolerance $(tol)")

    # Whitening matrix
    if dims == 1
        Vᵀ = UDVᵀ.Vt
        Wᵀ = broadcast!(/, Vᵀ, Vᵀ, D)  # in-place diagonal matrix multiply DVᵀ
    else
        U = UDVᵀ.U
        Wᵀ = broadcast!(/, U, U, transpose(D))
    end

    return (copy(transpose(Wᵀ)), detΣ)
end


function whiten_cov!(Σ::AbstractMatrix{T}, γ::T=zero(T)) where T
    (p = size(Σ, 1)) == size(Σ, 2) || throw(DimensionMismatch("Σ must be square"))

    0 ≤ γ ≤ 1 || throw(DomainError(γ, "γ must be in the interval [0,1]"))
    
    if γ != 0
        regularize!(Σ, γ)
    end
    
    W = inv(cholesky!(Σ, Val(false); check=true).U)
end

