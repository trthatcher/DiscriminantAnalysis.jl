


# Linear Discriminant Analysis

mutable struct LinearDiscriminantModel{T} <: DiscriminantModel{T}
    "Model fit indicator - `true` if model has been fit"
    fit::Bool
    "Dimension along which observations are stored (1 for rows, 2 for columns)"
    dims::Int
    "Whitening transform for overall covariance matrix"
    W::AbstractMatrix{T}
    "Determinant of the covariance matrix"
    detΣ::T
    "Matrix of class centroids (one per row or column - see `dims`)"
    M::Matrix{T}
    "Prior-weighted overall centroid"
    μ::Vector{T}
    "Vector of class prior probabilities"
    π::Vector{T}
    "Vector of Class counts"
    nₘ::Vector{Int}
    "Matrix of canonical coordinates"
    C::Union{Nothing,Matrix{T}}
    "Matrix of canonical coordinates with whitening applied"
    A::Union{Nothing,Matrix{T}}
    "Shrinkage parameter"
    γ::Union{Nothing,T}
    function LinearDiscriminantModel{T}() where T
        new{T}(false, 0, Array{T}(undef,0,0), zero(T), Array{T}(undef,0,0), 
               Array{T}(undef,0), Array{T}(undef,0), Array{Int}(undef,0), nothing, nothing, 
               nothing)
    end
end


function canonical_coordinates!(LDA::LinearDiscriminantModel{T}) where T
    m, p = check_dims(LDA.M, dims=LDA.dims)

    if p ≤ m-1
        # no dimensionality reduction is possible
        LDA.C = Matrix{Float64}(I, p, p)
        LDA.A = copy(LDA.W)
    elseif LDA.dims == 1
        # center M by overall centroid
        M = broadcast!(-, similar(LDA.M, T), LDA.M, transpose(LDA.μ))
        # Σ_between = Mᵀdiag(π)M so need to scale by sqrt π
        broadcast!((πₖ, Mₖⱼ) -> √(πₖ)Mₖⱼ, M, LDA.π, M)
        UDVᵀ = svd!(M*LDA.W, full=false)

        LDA.C = copy(transpose(view(UDVᵀ.Vt, 1:m-1, :)))
        LDA.A = LDA.W*LDA.C
    else
        M = broadcast!(-, similar(LDA.M, T), LDA.M, LDA.μ)
        broadcast!((πₖ, Mⱼₖ) -> √(πₖ)Mⱼₖ, M, transpose(LDA.π), M)
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
    n, p = check_dims(X, dims=dims)
    m = maximum(y)
    
    n₂ = length(y)
    n₂ == n || throw(DimensionMismatch("observation count along length of class index " *
                                       "vector y must match dimension $(dims) of data " *
                                       "matrix X (got $(n₂) and $(n))"))

    LDA.dims = dims
    is_row = dims == 1
    altdims = is_row ? 2 : 1

    if gamma !== nothing
        0 ≤ gamma ≤ 1 || throw(DomainError(gamma, "γ must be in the interval [0,1]"))
    end
    LDA.γ = gamma

    # Compute centroids and class counts from data if not specified
    if centroids === nothing
        LDA.M = is_row ? Matrix{T}(undef, m, p) : Matrix{T}(undef, p, m)
        LDA.nₘ = Vector{Int}(undef, m)

        class_statistics!(LDA.M, LDA.nₘ, X, y, dims=dims)
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

        LDA.M = copyto!(similar(centroids, T), centroids)
        LDA.nₘ = class_counts!(Vector{Int}(undef, m), y)
    end

    validate_class_counts(LDA.nₘ)

    # Compute priors from class frequencies in data if not specified
    if priors === nothing
        LDA.π = broadcast!(/, Vector{T}(undef, m), LDA.nₘ, n)
    else
        m₃ = length(priors)
        if m₃ != m
            throw(DimensionMismatch("class count along length of class prior probability " *
                                    "vector π must match maximum class index found in " *
                                    "class index vector y (got $(m₃) and $(m))"))
        end
        validate_priors(priors)

        LDA.π = copyto!(similar(priors, T), priors)
    end

    # Overall centroid is prior-weighted average of class centroids
    LDA.μ = is_row ? vec(transpose(LDA.π)*LDA.M) : LDA.M*LDA.π

    # Center the data matrix with respect to classes to compute whitening matrix
    X .-= is_row ? view(LDA.M, y, :) : view(LDA.M, :, y)

    # Use cholesky whitening if gamma is not specifed, otherwise svd whitening
    if LDA.γ === nothing
        LDA.W, LDA.detΣ = whiten_data!(X, dims=dims, df=n-m)
    else
        LDA.W, LDA.detΣ = whiten_data!(X, LDA.γ, dims=dims, df=n-m)
    end

    # Perform canonical discriminant analysis if applicable
    if canonical
        canonical_coordinates!(LDA)
    else
        LDA.C = nothing
        LDA.A = nothing
    end

    LDA.fit = true

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

#function discriminants!(LDA::LinearDiscriminantModel{T}, X::Matrix{T}; 
#                        dims::Integer=1) where T
#    Δ = Array{T}(undef, dims == 1 ? )
