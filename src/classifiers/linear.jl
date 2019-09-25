# Linear Discriminant Analysis

mutable struct LinearDiscriminantModel{T} <: DiscriminantModel{T}
    "Model fit indicator - `true` if model has been fit"
    Θ::DiscriminantParameters{T}
    "Whitening transformation"
    W::AbstractMatrix{T}
    "Matrix of canonical coordinates"
    C::Union{Nothing,AbstractMatrix{T}}
    "Discriminant function intercept"
    δ₀::T
    function LinearDiscriminantModel{T}() where T
        new{T}(DiscriminantParameters{T}(), Matrix{T}(undef,0,0), nothing, zero(T))
    end
end


function canonical_coordinates!(LDA::LinearDiscriminantModel{T}) where T
    Θ = LDA.Θ

    m, p = check_dims(Θ.M, dims=Θ.dims)

    if p ≤ m-1
        # no dimensionality reduction is possible
        LDA.C = Matrix{T}(I, p, p)
    elseif Θ.dims == 1
        # center M by overall centroid
        M = broadcast!(-, similar(Θ.M, T), Θ.M, transpose(Θ.μ))
        # Σ_between = Mᵀdiag(π)M so need to scale by sqrt π
        broadcast!((πₖ, Mₖⱼ) -> √(πₖ)Mₖⱼ, M, Θ.π, M)
        UDVᵀ = svd!(M*LDA.W, full=false)
        # Extract m-1 components 
        LDA.C = LDA.W*transpose(view(UDVᵀ.Vt, 1:m-1, :))
    else
        M = broadcast!(-, similar(Θ.M, T), Θ.M, Θ.μ)
        broadcast!((πₖ, Mⱼₖ) -> √(πₖ)Mⱼₖ, M, transpose(Θ.π), M)
        UDVᵀ = svd!(LDA.W*M, full=false)

        LDA.C = transpose(view(UDVᵀ.U, :, 1:m-1))*LDA.W
    end

    return LDA
end


"""
    fit!(LDA::LinearDiscriminantModel, Y::Matrix, X::Matrix; dims::Integer=1, kwargs...)

Fit a linear discriminant model based on data matrix `X` and class indicator matrix `Y` 
along dimensions `dims` and overwrite `LDA`.

# Keyword arguments
- `canonical::Bool=false`: computes canonical coordinates and performs dimensionality
reduction if `true`
- `compute_covariance`: if true, computes `Σ`
- `centroids::Matrix`: specifies the class centroids. If `dims` is `1`, then each row 
represents a class centroid, otherwise each column represents a class centroid. If not 
specified, the centroids are computed from the data.
- `priors::Vector`: specifies the class membership prior probabilties. If not specified, 
the class probabilities a computed based on the class counts
- `gamma::Real=0`: regularization parameter γ ∈ [0,1] shrinks the within-class covariance 
matrix towards the identity matrix scaled by the average eigenvalue of the covariance matrix
"""
function fit!(LDA::LinearDiscriminantModel{T},
              y::Vector{<:Integer},
              X::Matrix{T};
              dims::Integer=1,
              canonical::Bool=false,
              compute_covariance::Bool=false,
              centroids::Union{Nothing,AbstractMatrix}=nothing, 
              priors::Union{Nothing,AbstractVector}=nothing,
              gamma::Union{Nothing,Real}=nothing) where T
    Θ = LDA.Θ
    
    set_dimensions!(Θ, y, X, dims)
    set_gamma!(Θ, gamma)
    set_statistics!(Θ, y, X, centroids)
    set_priors!(Θ, priors)

    # Compute degress of freedom
    df = size(X, dims) - Θ.m

    if dims == 1
        # Overall centroid is prior-weighted average of class centroids
        Θ.μ = vec(transpose(Θ.π)*Θ.M)
        # Center the data matrix with respect to classes to compute whitening matrix
        X .-= view(Θ.M, y, :)
        # Compute covariance matrix if requested
        if compute_covariance
            Θ.Σ = lmul!(one(T)/df, transpose(X)*X)
        end
    else
        Θ.μ = Θ.M*Θ.π
        X .-= view(Θ.M, :, y)
        if compute_covariance
            Θ.Σ = lmul!(one(T)/df, X*transpose(X))
        end
    end

    # Use cholesky whitening if gamma is not specifed, otherwise svd whitening
    if Θ.γ === nothing
        LDA.W, LDA.δ₀ = whiten_data!(X, dims=dims, df=df)
    else
        LDA.W, LDA.δ₀ = whiten_data!(X, Θ.γ, dims=dims, df=df)
    end

    # Perform canonical discriminant analysis if applicable
    if canonical
        canonical_coordinates!(LDA)
    else
        LDA.C = nothing
    end

    Θ.fit = true

    return LDA
end


function discriminants!(Δ::Matrix{T}, LDA::LinearDiscriminantModel{T}, X::Matrix{T}) where T
    dims = LDA.Θ.dims

    M = LDA.Θ.M

    n, p = check_dims(X, dims=dims)
    m, p₂ = check_dims(M, dims=dims)
    n₂, m₂ = check_dims(Δ, dims=dims)

    alt_dims = dims == 1 ? 2 : 1

    p == p₂ || throw(DimensionMismatch("predictor count along dimension $(alt_dims) of " *
                                       "X must match dimension $(dims) of M (got $(p) " *
                                       "and $(p₂))"))
    n == n₂ || throw(DimensionMismatch("observation count along dimension $(dims) of X " *
                                       "must match dimension $(dims) of Δ (got $(n) " *
                                       "and $(n₂))"))
    m == m₂ || throw(DimensionMismatch("class count along dimension $(alt_dims) of Δ " *
                                       "must match dimension $(dims) of M (got $(m) " *
                                       "and $(m₂))"))

    is_row = dims == 1

    π = LDA.Θ.π
    W = LDA.C === nothing ? LDA.W : LDA.C

    X̃, M̃ = is_row ? (X*W, M*W) : (W*X, W*M)

    Z = similar(X̃)
    for k = 1:m
        if is_row
            Δₖ = view(Δ, :, k:k)
            μₖ = view(M̃, k:k, :)
        else
            Δₖ = view(Δ, k:k, :)
            μₖ = view(M̃, :, k:k)
        end

        broadcast!(-, Z, X̃, μₖ)
        sum!(abs2, Δₖ, Z)
        broadcast!((d², logπ) -> logπ - d²/2, Δₖ, Δₖ, log(π[k]))
    end

    return Δ
end