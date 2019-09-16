# Linear Discriminant Analysis

mutable struct LinearDiscriminantModel{T} <: DiscriminantModel{T}
    "Model fit indicator"
    fit::Bool
    "Dimension along which observations are stored (1 for rows, 2 for columns)"
    dims::Int
    "Whitening transform for overall covariance matrix"
    W::AbstractMatrix{T}
    "Determinant of the covariance matrix"
    detΣ::T
    "Matrix of class means (one per row/column depending on `dims`)"
    M::Matrix{T}
    "Overall centroid"
    μ::Vector{T}
    "Vector of class prior probabilities"
    π::Vector{T}
    "Class counts"
    nₘ::Vector{Int}
    "Canonical coordinates"
    C::Union{Nothing,Matrix{T}}
    "Canonical coordinates with whitening"
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

function _fit!(LDA::LinearDiscriminantModel{T},
               y::Vector{<:Integer},
               X::Matrix{T},
               dims::Integer=1,
               canonical::Bool=false,
               centroids::Union{Nothing,AbstractMatrix}=nothing, 
               priors::Union{Nothing,AbstractVector}=nothing,
               gamma::Union{Nothing,Real}=nothing) where T
    n, p = check_data_dims(X, y, dims=dims)
    m = maximum(y)

    LDA.dims = dims
    is_row = dims == 1

    if gamma !== nothing
        0 ≤ gamma ≤ 1 || throw(DomainError(gamma, "γ must be in the interval [0,1]"))
    end
    LDA.γ = gamma

    # Compute class counts for hypothesis tests
    LDA.nₘ = class_counts!(zeros(Int, m), y)
    all(LDA.nₘ .≥ 2) || error("must have at least two observations per class")

    # Compute priors from class frequencies in data if not specified
    if priors === nothing
        LDA.π = broadcast!(/, Vector{T}(undef, m), LDA.nₘ, n)
    else
        LDA.π = copyto!(similar(priors, T), priors)
    end
    check_priors(LDA.π)

    # Compute centroids from data if not specified
    if centroids === nothing
        LDA.M = is_row ? zeros(T, m, p) : zeros(T, p, m)
        class_centroids!(LDA.M, X, y, dims=dims)
    else
        check_centroid_dims(centroids, X, dims=dims)
        LDA.M = copyto!(similar(centroids, T), centroids)
    end
    if size(LDA.M, dims) != m
        error("here error")
    end

    # Overall centroid is prior-weighted average of class centroids
    LDA.μ = is_row ? vec(transpose(LDA.π)*LDA.M) : LDA.M*LDA.π

    # Center the data matrix with respect to classes to compute whitening matrix
    X .-= is_row ? view(LDA.M, y, :) : view(LDA.M, :, y)
    #center_classes!(X, LDA.M, y, dims=dims)

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


"""
    fit!(LDA::LinearDiscriminantModel, y::Vector, X::Matrix)

Fit a linear discriminant model based on data matrix `X` and class index vector `y`. `LDA` 
will be updated in-place and `X` will be overwritten during the fitting process.
"""
function fit!(LDA::LinearDiscriminantModel{T}, y::Vector{<:Integer}, X::Matrix{T}) where T
    center_classes!(X, y, LDA.M, LDA.dims)

    if LDA.γ === nothing
        copyto!(LDA.W, whiten_data!(X, LDA.dims))
    else
        copyto!(LDA.W, whiten_data!(X, LDA.γ, LDA.dims))
    end
    
    if LDA.C !== nothing
        M = copy(LDA.M)
        μ = LDA.dims == 1 ? transpose(LDA.π)*M : M*LDA.π
        broadcast!(-, M, M, μ)
        copyto!(LDA.C, canonical_coordinates(M, LDA.W, LDA.dims))
    end

    LDA.fit = true

    return LDA
end

"""
    fit(::Type{LinearDiscriminantModel}, Y::Matrix, X::Matrix; dims::Integer=1, <keyword arguments>...)

Fit a linear discriminant model based on data matrix `X` and class indicator matrix `Y` 
along dimensions `dims`.

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
function fit(::Type{LinearDiscriminantModel{T}},
        Y::AbstractMatrix,
        X::AbstractMatrix;
        dims::Integer=1,
        canonical::Bool=false,
        centroids::Union{Nothing,AbstractMatrix}=nothing, 
        priors::Union{Nothing,AbstractVector}=nothing,
        gamma::Union{Nothing,Real}=nothing
    ) where T<:AbstractFloat

    k = size(Y, dims == 1 ? 2 : 1)
    p = size(X, dims == 1 ? 2 : 1)

    y = vec(mapslices(argmax, Y; dims=dims == 1 ? 2 : 1))

    if centroids === nothing
        M = class_means(X, y, dims, k)
    else
        M = copyto!(similar(centroids, T), centroids)
    end

    if priors === nothing
        nₖ = class_counts(y, k)

        all(nₖ .≥ 2) || error("must have at least two observations per class")

        π = ones(T, k)
        broadcast!(/, π, π, nₖ)
    else
        π = copyto!(similar(priors, T), priors)
    end

    LDA = LinearDiscriminantModel{T}(M, π, dims, gamma, canonical)

    fit!(LDA, y, X)
end

function fit(::Type{LinearDiscriminantModel},
        Y::AbstractMatrix,
        X::AbstractMatrix{T};
        dims::Integer=1,
        canonical::Bool=false,
        centroids::Union{Nothing,AbstractMatrix}=nothing, 
        priors::Union{Nothing,AbstractVector}=nothing,
        gamma::Union{Nothing,Real}=nothing
    ) where T
    fit(LinearDiscriminantModel{T}, Y, X, dims=dims, canonical=canonical, 
        centroids=centroids, priors=priors, gamma=gamma)
end

function discriminants!(Δ::Matrix{T}, LDA::LinearDiscriminantModel{T}, X::Matrix{T}; 
                        dims::Integer=1) where T
    dims ∈ (1, 2) || arg_error("dims should be 1 or 2 (got $(dims))")

    M = LDA.M
    W = LDA.W

    if dims == 1
        k, p = size(M)
        n = size(Δ, 1)

        size(Δ, 2) == k || dim_error("the number of columns in discriminant matrix Δ " *
                                     "must match the number of classes")

        size(X, 2) == p || dim_error("the number of columns in data matrix X must match " *
                                     "the number of columns in centroid matrix M")

        size(X, 1) == n || dim_error("the number of rows in data matrix X must match the " *
                                     "number of rows in discriminant matrix Δ")

        XW = X*W
        MW = M*W
        Z = similar(XW)
        for j = 1:k
            broadcast!(-, Z, XW, MW[j, :])
            sum!(abs2, Δ[:, j], Z)
        end
    else
        p, k = size(M)
        n = size(Δ, 2)

        size(Δ, 1) == k || dim_error("")

        size(X, 1) == p || dim_error("")

        size(X, 2) == n || dim_error("")

        WX = W*X
        WM = W*M
        Z = similar(XW)
        for i = 1:k
            broadcast!(-, Z, WX, WM[:, i])
            sum!(abs2, Δ[i, :], Z)
        end
    end

    return Δ
end

#function discriminants!(LDA::LinearDiscriminantModel{T}, X::Matrix{T}; 
#                        dims::Integer=1) where T
#    Δ = Array{T}(undef, dims == 1 ? )
