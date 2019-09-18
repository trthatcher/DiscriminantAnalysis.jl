# Linear Discriminant Analysis

mutable struct LinearDiscriminantModel{T} <: DiscriminantModel{T}
    "Model fit indicator - `true` if model has been fit"
    Θ::DiscriminantParameters{T}
    "Whitening transformation"
    W::Matrix{T}
    "Matrix of canonical coordinates"
    C::Union{Nothing,Matrix{T}}
    "Matrix of canonical coordinates with whitening applied"
    A::Union{Nothing,Matrix{T}}
    function LinearDiscriminantModel{T}() where T
        new{T}(DiscriminantParameters{T}(), Matrix{T}(undef,0,0), nothing, nothing)
    end
end


function canonical_coordinates!(LDA::LinearDiscriminantModel{T}) where T
    Θ = LDA.Θ

    m, p = check_dims(Θ.M, dims=Θ.dims)

    if p ≤ m-1
        # no dimensionality reduction is possible
        LDA.C = Matrix{Float64}(I, p, p)
        LDA.A = copy(LDA.W)
    elseif Θ.dims == 1
        # center M by overall centroid
        M = broadcast!(-, similar(Θ.M, T), Θ.M, transpose(Θ.μ))
        # Σ_between = Mᵀdiag(π)M so need to scale by sqrt π
        broadcast!((πₖ, Mₖⱼ) -> √(πₖ)Mₖⱼ, M, Θ.π, M)
        UDVᵀ = svd!(M*LDA.W, full=false)

        LDA.C = copy(transpose(view(UDVᵀ.Vt, 1:m-1, :)))
        LDA.A = LDA.W*LDA.C
    else
        M = broadcast!(-, similar(Θ.M, T), Θ.M, Θ.μ)
        broadcast!((πₖ, Mⱼₖ) -> √(πₖ)Mⱼₖ, M, transpose(Θ.π), M)
        UDVᵀ = svd!(LDA.W*M, full=false)

        LDA.C = copy(transpose(view(UDVᵀ.U, :, 1:m-1)))
        LDA.A = LDA.C*LDA.W
    end

    return LDA
end


"""
    fit!(LDA::LinearDiscriminantModel, Y::Matrix, X::Matrix; dims::Integer=1, <keyword arguments>...)

Fit a linear discriminant model based on data matrix `X` and class indicator matrix `Y` 
along dimensions `dims` and overwrite `LDA`.

# Keyword arguments
- `canonical::Bool=false`: computes canonical coordinates and performs dimensionality
reduction if `true`
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
              X::Matrix{T},
              dims::Integer=1,
              canonical::Bool=false,
              centroids::Union{Nothing,AbstractMatrix}=nothing, 
              priors::Union{Nothing,AbstractVector}=nothing,
              gamma::Union{Nothing,Real}=nothing) where T
    Θ = LDA.Θ
    parameter_fit!(Θ, y, X, dims, false, centroids, priors, gamma)

    df = size(X, dims) - size(Θ.M, dims)

    # Use cholesky whitening if gamma is not specifed, otherwise svd whitening
    if Θ.γ === nothing
        LDA.W, Θ.detΣ = whiten_data!(X, dims=dims, df=df)
    else
        LDA.W, Θ.detΣ = whiten_data!(X, Θ.γ, dims=dims, df=df)
    end

    # Perform canonical discriminant analysis if applicable
    if canonical
        canonical_coordinates!(LDA)
    else
        LDA.C = nothing
        LDA.A = nothing
    end

    Θ.fit = true

    return LDA
end


function discriminants!(Δ::Matrix{T}, LDA::LinearDiscriminantModel{T}, X::Matrix{T}; 
                        dims::Integer=1) where T
    n, p = check_dims(X, dims=dims)
    m, p₂ = check_dims(LDA.M, dims=dims)
    n₂, m₂ = check_dims(Δ)

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

    M = LDA.M
    W = LDA.W

    if dims == 1
        XW = X*W
        MW = M*W
        Z = similar(XW)
        for k = 1:m
            broadcast!(-, Z, XW, MW[k, :])
            sum!(abs2, Δ[:, k], Z)
        end
    else
        WX = W*X
        WM = W*M
        Z = similar(XW)
        for k = 1:m
            broadcast!(-, Z, WX, WM[:, k])
            sum!(abs2, Δ[k, :], Z)
        end
    end

    return Δ
end
