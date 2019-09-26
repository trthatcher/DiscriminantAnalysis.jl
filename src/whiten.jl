### Data Whitening Functions

"""
    whiten_data!(X; dims, df)

Compute a whitening transform matrix for centered data matrix `X`. Use `dims=1` for 
row-based observations and `dims=2` for column-based observations. The `df` parameter 
specifies the effective degrees of freedom.
"""
function whiten_data_chol!(X::Matrix{T}; dims::Integer, df::Integer=size(X,dims)-1) where T
    df > 0 || error("degrees of freedom must be greater than 0")

    n, p = check_dims(X, dims=dims)

    n > p || error("insufficient number of within-class observations to produce a full " *
                   "rank covariance matrix ($(n) observations, $(p) predictors)")

    if dims == 1
        # X = QR ⟹ S = XᵀX = RᵀR
        W⁻¹ = UpperTriangular(qr!(X, Val(false)).R)  
    else
        # Xᵀ = LQ ⟹ S = XXᵀ = LLᵀ = RᵀR
        W⁻¹ = LowerTriangular(lq!(X).L)
    end

    broadcast!(/, W⁻¹, W⁻¹, √(df))

    detΣ = det(W⁻¹)^2

    W = try
        inv(W⁻¹)
    catch err
        if isa(err, LAPACKException) || isa(err, SingularException)
            if err.info ≥ 1
                error("rank deficiency detected (collinearity in predictors)")
            end
        end
        throw(err)
    end

    return (W, detΣ)
end


@inline regularize(x, y, γ) = (1-γ)*x + γ*y


function whiten_data_svd!(X::Matrix{T}, γ::Union{Nothing,T}; dims::Integer, 
                          df::Integer=size(X,dims)-1) where T
    n, p = check_dims(X, dims=dims)
    
    n > p || error("insufficient number of within-class observations to produce a full " *
                   "rank covariance matrix ($(n) observations, $(p) predictors)")
    
    0 ≤ γ ≤ 1 || throw(DomainError(γ, "γ must be in the interval [0,1]"))

    tol = eps(one(T))*p*maximum(X)

    UDVᵀ = svd!(X, full=false)

    D = UDVᵀ.S

    if γ !== nothing && γ ≠ zero(T)
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


function whiten_cov_chol!(Σ::AbstractMatrix{T}, γ::Union{Nothing,T}=zero(T); 
                          dims::Integer=1) where T
    p, p₂ = check_dims(Σ, dims=dims)

    p == p₂ || throw(DimensionMismatch("Σ must be square"))

    if γ !== nothing && γ ≠ zero(T)
        zero(T) ≤ γ ≤ one(T) || throw(DomainError(γ, "γ must be in the interval [0,1]"))
        regularize!(Σ, γ)
    end

    UᵀU = cholesky!(Σ, Val(false); check=true)
    
    if dims == 1
        Lᵀ = UᵀU.U
        detΣ = det(Lᵀ)^2

        return (inv(Lᵀ), detΣ)
    else
        L = UᵀU.L
        detΣ = det(L)^2

        return (inv(L), detΣ)
    end 
end