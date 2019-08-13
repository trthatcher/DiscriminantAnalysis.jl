# Error shortcuts
dim_error(s) = throw(DimensionMismatch(s))
arg_error(s) = throw(ArgumentError(s))

function _class_means!(M::AbstractMatrix{T}, X::AbstractMatrix{T}, y::Vector{<:Int}) where T
    n, p = size(X)
    k, pₘ = size(M)
    l = length(y)
    
    p == pₘ || dim_error("the number of columns in X must match the number of columns in " *
                         "M (got $(p) and $(pₘ))")
    
    n == l || dim_error("the number of rows in X must match the length y (got $(n) and " *
                        "$(l))")
           
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

    all(nₖ .> 1) || error("must have at least two observations per class")
    
    return broadcast!(/, M, M, nₖ)
end

function _class_means(X::AbstractMatrix{T}, y::Vector{<:Int}; dims::Integer=1, 
                      k::Integer=maximum(y)) where T
    dims ∈ (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $(dims))"))

    if dims == 1  # rows
        M = Array{T}(undef, k, size(X, 2))
        return _class_means!(M, X, y)
    else  # columns
        p, n = size(X)
        l = length(y)

        n == l || dim_error("the number of columns in X must match the length y (got " * 
                            "$(n) and $(l))")
        
        M = Array{T}(undef, p, k)
        _class_means!(transpose(M), transpose(X), y)
        return M
    end
end


"""
_center_classes!(X, y, M)
"""
function _center_classes!(X::AbstractMatrix, y::Vector{<:Int}, M::AbstractMatrix)
    n, p = size(X)
    k, pₘ = size(M)
    l = length(y)

    p == pₘ || dim_error("the number of columns in X must match the number of columns in " *
                         "M (got $(p) and $(pₘ))")
    
    n == l || dim_error("the number of rows in X must match the length y (got $(n) and " *
                        "$(l))")
    
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
function _center_classes!(X::AbstractMatrix, y::Vector{<:Int}, M::AbstractMatrix, 
                          dims::Integer)
    dims ∈ (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))

    if dims == 1
        return _center_classes!(X, y, M)
    else
        p, n = size(X)
        pₘ, k = size(M)
        l = length(y)

        p == pₘ || dim_error("the number of rows in X must match the number of rows in M " *
                             "(got $(p) and $(pₘ))")

        n == l || dim_error("the number of columns in X must match the length y (got " * 
                            "$(n) and $(l))")

        _center_classes!(transpose(X), y, transpose(M))
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
function _whiten_data!(X::Matrix{T}, dims::Int) where T
    dims ∈ (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    n = size(X, dims)
    p = size(X, dims == 1 ? 2 : 1)
    n > p || error("insufficient number of within-class observations to produce a full " *
                   "rank covariance matrix ($(n) observation, $(p) predictors)")

    R = if dims == 1
        # X = QR ⟹ S = XᵀX = RᵀR
        UpperTriangular(qr!(X, Val(false)).R)  
    else
        # Xᵀ = LQ ⟹ S = XXᵀ = LLᵀ = RᵀR
        UpperTriangular(transpose(lq!(X).L))  
    end

    W = try
        inv(R)
    catch err
        if isa(err, LAPACKException) || isa(err, SingularException)
            if err.info ≥ 1
                error("rank deficiency (collinearity) detected")
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


# transpose(X*W')*(X*W')
function _whiten_data(X::AbstractMatrix{T}, γ::T, ϵ::T=size(X,2)*eps(T)*maximum(X)) where T
    0 ≤ γ ≤ 1 || error("γ must be in the interval [0,1] (got $(γ))")

    n, p = size(X)
    n > p || error("insufficient number of within-class observations to produce a full " *
                   "rank covariance matrix ($(n) observation, $(p) predictors)")

    UDVᵀ = svd(X, full=false)  # Vᵀ from thin SVD will be n×n since m > n

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

