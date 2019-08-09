"""
    regularize(Σ₁, Σ₂, λ)
"""
function regularize!(Σ₁::Matrix{T}, Σ₂::Matrix{T}, λ::T) where {T}
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
    regularize(Σ, γ)

Shrink `Σ` matrix towards the average eigenvalue multiplied by the identity matrix
"""
function regularize!(Σ::Matrix{T}, γ::T) where {T}
    m, n = size(Σ)
    m == n || throw(DimensionMismatch("matrices Σ₁ and Σ₂ must be square"))
    0 ≤ γ ≤ 1 || throw(DomainError(γ, "γ must be in the interval [0,1]"))

    λ_avg = tr(Σ)/m  # Find average eigenvalue

    Σ *= 1 - γ
    Σ += (γ*λ_avg)*I

    return Σ
end

"""
    _center_classes!(X, y, M)
"""
function _center_classes!(X::AbstractMatrix, y::Vector{<:Int}, M::AbstractMatrix)
    m, n = size(X)
    k = size(M, 1)

    for i = 1:m
        kᵢ = y[i]
        1 ≤ kᵢ ≤ k || throw(BoundsError(M, (kᵢ, 1)))
        @inbounds for j = 1:n
            X[i, j] -= M[kᵢ, j]
        end
    end

    return X
end

"""
    center_classes!(X, y, M)
"""
function center_classes!(X::AbstractMatrix, y::Vector{<:Int}, M::AbstractMatrix, 
                         dims::Integer=2)
    dims ∈ (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))

    mX, nX = size(X)
    ly = length(y)
    if dims == 1
        nM = size(M, 2)
        mX == ly || throw(DimensionMismatch("the number of rows in X must match the " *
                                            "length of y (got $mX and $ly"))
        nX == nM || throw(DimensionMismatch("the number of columns in X must match the " *
                                            "number of columns in M (got $nX and $nM"))
        _center_classes!(X, y, M)
    else
        mM = size(M, 1)
        nX == ly || throw(DimensionMismatch("the number of columns in X must match the " *
                                            "length of y (got $nX and $ly"))
        mX == mM || throw(DimensionMismatch("the number of rows in X must match the " *
                                            "number of rows in M (got $mX and $mM"))
        _center_classes!(transpose(X), y, transpose(M))
    end
end