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

Shrink `Σ` matrix towards the average eigenvalue multiplied by the identity matrix
"""
function regularize!(Σ::AbstractMatrix{T}, γ::T) where {T}
    (p = size(Σ, 1)) == size(Σ, 2) || throw(DimensionMismatch("Σ must be square"))
    0 ≤ γ ≤ 1 || throw(DomainError(γ, "γ=$(γ) must be in the interval [0,1]"))

    a =  γ*tr(Σ)/p  # Average eigenvalue scaled by γ
    
    broadcast!(*, Σ, Σ, 1 - γ)
    for i = 1:p
        Σ[i, i] += a
    end

    return Σ
end


"""
_whiten_data!(X::Matrix{T})

Generate whitening transform matrix for centered data matrix X
"""
function _whiten_data!(X::AbstractMatrix{T}) where T
    n, p = size(X)
    n > p || error("insufficient number of within-class observations to produce a full " *
                   "rank covariance matrix ($(n) observation, $(p) predictors)")

    R = UpperTriangular(qr!(X, Val(false)).R)  # X = QR ⟹ S = X'X = R'R
    
    try
        W = inv(R)
        broadcast!(*, W, W, √(n-1))
    catch err
        if isa(err, LAPACKException)
            if err.info ≥ 1
                error("rank deficiency (collinearity) detected")
            end
        end
        throw(err)
    end
end


# transpose(X*W')*(X*W')
function _whiten_data!(X::AbstractMatrix{T}, γ::T, ϵ::T=size(X,2)*eps(T)*maximum(X)) where T
    0 ≤ γ ≤ 1 || error("γ must be in the interval [0,1] (got $(γ))")

    n, p = size(X)
    n > p || error("insufficient number of within-class observations to produce a full " *
                   "rank covariance matrix ($(n) observation, $(p) predictors)")

    UDVᵀ = svd!(X, full=false)  # Vᵀ from thin SVD will be n×n since m > n

    D = UDVᵀ.S
    broadcast!(σᵢ -> (σᵢ^2)/(n-1), D, D)  # Convert data singular values to cov eigenvalues

    # Regularize: Σ = VD²Vᵀ ⟹ Σ(γ) = V((1-γ)D² + (γ/p)trace(D²)I)Vᵀ
    if γ ≠ 0
        λ_bar = mean(D)
        broadcast!(λᵢ -> √((1-γ)*λᵢ + γ*λ_bar), D, D)
    else
        broadcast!(√, D, D)
    end

    all(D .≥ ϵ) || error("rank deficiency (collinearity) detected with tolerance $(ϵ)")

    # Whitening matrix
    Vᵀ = UDVᵀ.Vt
    W = broadcast!(/, Vᵀ, Vᵀ, D)
end


function whiten_cov!(Σ::AbstractMatrix{T}, γ::T=zero(T)) where T
    (p = size(Σ, 1)) == size(Σ, 2) || throw(DimensionMismatch("Σ must be square"))
    0 ≤ γ ≤ 1 || error("γ must be in the interval [0,1] (got $(γ))")
    
    if γ != 0
        regularize!(Σ, γ)
    end
    
    W = inv(cholesky!(Σ, Val(false); check=true).U)
end

