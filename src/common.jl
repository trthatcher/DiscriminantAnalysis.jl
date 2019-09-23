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
        nₖ > 1 || error("class count at class index $(k) must be greater than 1")
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
    p, n = size(X)
    p₂, m = size(M)
    n₂ = length(y)
    m₂ = length(nₘ)

    p == p₂ || throw(DimensionMismatch("predictor count mismatch between M and X"))
    n == n₂ || throw(DimensionMismatch("observation count mismatch between y and X"))
    m == m₂ || throw(DimensionMismatch("class count mismatch between nₘ and M"))

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
    n, p = check_dims(X, dims=dims)
    m, p₂ = check_dims(M, dims=dims)
    n₂ = length(y)
    m₂ = length(nₘ)

    altdims = dims == 1 ? 2 : 1
    
    p == p₂ || throw(DimensionMismatch("predictor count along dimension $(altdims) of X " *
                                       "must match dimension $(altdims) of M (got $(p) " * 
                                       "and $(p₂))"))
    n == n₂ || throw(DimensionMismatch("observation count along length of y must match " *
                                       "dimension $(dims) of X (got $(n) and $(n₂))"))
    m == m₂ || throw(DimensionMismatch("class count along length of nₘ must match " *
                                       "dimension $(dims) of M (got $(m₂) and $(m))"))

    if dims == 1
        _class_statistics!(transpose(M), nₘ, transpose(X), y)
        return (M, nₘ)
    else dims == 2
        return _class_statistics!(M, nₘ, X, y)
    end
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