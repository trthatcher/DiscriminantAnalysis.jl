# Common Functions

### Dimensionality Checks

function check_dims(X::AbstractMatrix, dims::Integer)
    dims ∈ (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))

    n = size(X, dims)
    p = size(X, dims == 1 ? 2 : 1)

    return (n, p)
end

function check_centroid_dims(M::AbstractMatrix, X::AbstractMatrix, dims::Integer)
    n, p = check_dims(X, dims)
    k, pₘ = check_dims(M, dims)
    
    if p != pₘ
        rc = dims == 1 ? "columns" : "rows"
        msg = "the number of $(rc) in centroid matrix M must match the number of $(rc) " *
              "in data matrix X (got $(pₘ) and $(p))"

        throw(DimensionMismatch(msg))
    end

    return (n, p, k)
end

function check_centroid_dims(M::AbstractMatrix, π::AbstractVector, dims::Integer)
    k, p = check_dims(M, dims)
    l = length(π)

    if k != l
        rc = dims == 1 ? "columns" : "rows"
        msg = "the number of $(rc) in centroid matrix M must match the legth of class " *
              "index vector y (got $(k) and $(l))"

        throw(DimensionMismatch(msg))
    end

    return (k, p)
end

function check_data_dims(X::AbstractMatrix, y::AbstractVector, dims::Integer)
    n, p = check_dims(X, dims)
    l = length(y)

    if n != l
        rc = dims == 1 ? "rows" : "columns"
        msg = "the number of $(rc) in data matrix X must match the length of class index " *
              "vector y (got $(n) and $(l))"

        throw(DimensionMismatch(msg))
    end

    return n, p
end

function check_priors(π::Vector{T}) where T
    total = sum(π)

    if !isapprox(total, one(T))
        throw(ArgumentError("class priors π must sum to 1 (got $(total))"))
    end

    if any(π .≤ 0)
        throw(DomainError(π, "all class priors πₖ must be positive probabilities"))
    end

    return length(π)
end


### Class Calculations

function class_means!(M::AbstractMatrix{T}, X::AbstractMatrix{T}, y::Vector{<:Int}) where T
    n, p, k = check_centroid_dims(M, X, 1)
    check_data_dims(X, y, 1)
           
    M .= zero(T)
    nₖ = zeros(Int, k)  # track counts to ensure an observation for each class
    for i = 1:n
        kᵢ = y[i]
        1 ≤ kᵢ ≤ k || throw(BoundsError(M, (kᵢ, 1)))
        nₖ[kᵢ] += 1
        @inbounds for j = 1:p
            M[kᵢ, j] += X[i, j]
        end
    end

    all(nₖ .≥ 1) || error("must have at least one observation per class")
    broadcast!(/, M, M, nₖ)

    return M
end

function class_means(X::AbstractMatrix{T}, y::Vector{<:Integer}, dims::Integer=1, 
                     k::Integer=maximum(y)) where T
    if dims == 1
        return class_means!(zeros(T, k, size(X, 2)), X, y)
    else
        n, p = check_data_dims(X, y, dims)

        M = zeros(T, p, k)
        class_means!(transpose(M), transpose(X), y)

        return M
    end
end

function class_counts(y::Vector{<:Integer}, k::Integer=maximum(y))
    nₖ = zeros(Int, k)

    for i = 1:length(y)
        kᵢ = y[i]
        1 ≤ kᵢ ≤ k || throw(BoundsError(nₖ, kᵢ))
        nₖ[kᵢ] += 1
    end

    return nₖ
end


"""
_center_classes!(X, y, M)
"""
function center_classes!(X::AbstractMatrix, y::Vector{<:Int}, M::AbstractMatrix)
    n, p, k = check_centroid_dims(M, X, 1)
    check_data_dims(X, y, 1)
    
    for i = 1:n
        kᵢ = y[i]
        1 ≤ kᵢ ≤ k || throw(BoundsError(M, (kᵢ, 1)))
        @inbounds for j = 1:p
            X[i, j] -= M[kᵢ, j]
        end
    end

    return X
end


"""
_center_classes!(X, y, M, dims)
"""
function center_classes!(X::AbstractMatrix, y::Vector{<:Int}, M::AbstractMatrix, 
                         dims::Integer)
    if dims == 1
        return center_classes!(X, y, M)
    else
        n, p, k = check_centroid_dims(M, X, dims)
        check_data_dims(X, y, 2)

        center_classes!(transpose(X), y, transpose(M))

        return X
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

    0 ≤ γ ≤ 1 || throw(DomainError(γ, "γ must be in the interval [0,1]"))

    a =  γ*tr(Σ)/p  # Average eigenvalue scaled by γ
    
    broadcast!(*, Σ, Σ, 1 - γ)
    for i = 1:p
        Σ[i, i] += a
    end

    return Σ
end


"""
whiten_data!(X::Matrix{T})

Generate whitening transform matrix for centered data matrix X
"""
function whiten_data!(X::Matrix{T}, dims::Integer) where T
    n, p = check_dims(X, dims)

    n > p || error("insufficient number of within-class observations to produce a full " *
                   "rank covariance matrix ($(n) observations, $(p) predictors)")

    if dims == 1
        # X = QR ⟹ S = XᵀX = RᵀR
        R = UpperTriangular(qr!(X, Val(false)).R)  
    else
        # Xᵀ = LQ ⟹ S = XXᵀ = LLᵀ = RᵀR
        R = UpperTriangular(transpose(lq!(X).L))  
    end

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
    
    broadcast!(*, W, W, √(n-1))

    if dims == 1
        return W
    else
        return transpose(W)
    end
end


function whiten_data!(X::Matrix{T}, γ::T, dims::Integer; atol::T = zero(T),
                      rtol::T = (eps(one(T))*min(size(X)...))*iszero(atol)) where T
    n, p = check_dims(X, dims)
    
    n > p || error("insufficient number of within-class observations to produce a full " *
                   "rank covariance matrix ($(n) observations, $(p) predictors)")
    
    0 ≤ γ ≤ 1 || throw(DomainError(γ, "γ must be in the interval [0,1]"))


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

    tol = max(rtol*maximum(D), atol)
    all(D .> tol) || error("rank deficiency (collinearity) detected with tolerance $(tol)")

    # Whitening matrix
    if dims == 1
        Vᵀ = UDVᵀ.Vt
        Wᵀ = broadcast!(/, Vᵀ, Vᵀ, D)  # in-place diagonal matrix multiply DVᵀ
    else
        U = UDVᵀ.U
        Wᵀ = broadcast!(/, U, U, transpose(D))
    end

    return transpose(Wᵀ)
end


function whiten_cov!(Σ::AbstractMatrix{T}, γ::T=zero(T)) where T
    (p = size(Σ, 1)) == size(Σ, 2) || throw(DimensionMismatch("Σ must be square"))

    0 ≤ γ ≤ 1 || throw(DomainError(γ, "γ must be in the interval [0,1]"))
    
    if γ != 0
        regularize!(Σ, γ)
    end
    
    W = inv(cholesky!(Σ, Val(false); check=true).U)
end

