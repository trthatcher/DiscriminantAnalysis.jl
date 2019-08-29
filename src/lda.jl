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
    "Vector of class prior probabilities"
    π::Vector{T}
    "Canonical coordinates"
    C::Union{Nothing,Matrix{T}}
    "Canonical coordinates with whitening"
    A::Union{Nothing,Matrix{T}}
    "Shrinkage parameter"
    γ::Union{Nothing,T}
    function LinearDiscriminantModel{T}(M::AbstractMatrix, π::AbstractVector; 
                                        dims::Integer=1, canonical::Bool=false,
                                        gamma::Union{Nothing,Real}=nothing) where T
        k, p = check_centroid_dims(M, π, dims=dims)

        k ≥ 2 || error("must have at least two classes")

        check_priors(π)

        if gamma !== nothing
            0 ≤ gamma ≤ 1 || throw(DomainError(gamma, "γ must be in the interval [0,1]"))
        end

        W = zeros(T, p, p)

        if canonical
            d = min(k-1, p)
            C = dims == 1 ? zeros(T, p, d) : zeros(T, d, p)
            A = copy(C)
        else
            C = nothing
            A = nothing
        end

        new{T}(false, dims, W, zero(T), copy(M), copy(π), C, A, gamma)
    end
end

function canonical_coordinates!(LDA::LinearDiscriminantModel)
    M = copy(LDA.M)
    m, p = check_dims(M, dims=LDA.dims)

    π = LDA.π
    W = LDA.W

    if p ≤ m-1
        # no dimensionality reduction is possible
        C = I
    elseif LDA.dims == 1
        # Need to center M by overall mean
        # Σ = Mᵀdiag(π)M so need to scale by sqrt π
        M .-= transpose(π)*M
        broadcast!((a,b) -> a*√(b), M, M, π)
        UDVᵀ = svd!(M*W, full=false)
        C = transpose(view(UDVᵀ.Vt, 1:m-1, :))
    else
        M .-= M*π
        broadcast!((a,b) -> a*√(b), M, M, transpose(π))
        UDVᵀ = svd!(W*M, full=false)
        C = transpose(view(UDVᵀ.U, :, 1:m-1))
    end

    size(LDA.C) == size(C) || throw(DimensionMismatch("LDA.C does not match computed C"))
    copyto!(LDA.C, C)

    if p ≤ m-1
        size(LDA.A) == (p, p)
        copyto!(LDA.A, W)
    elseif LDA.dims == 1
        mul!(LDA.A, LDA.W, LDA.C)
    else
        mul!(LDA.A, LDA.C, LDA.W)
    end

    return LDA
end

#function canonical_coordinates(M::AbstractMatrix{T}, W::AbstractMatrix{T}, π::Vector{T},
#                               dims::Integer) where T
#    dims ∈ (1, 2) || arg_error("dims should be 1 or 2 (got $(dims))")
#    k = size(M, dims)
#    p = size(M, dims == 1 ? 2 : 1)
#
#    length(π) == k || dim_error("")
#
#    all(size(W) .== p) || dim_error("W must match dimensions of M")
#
#    d = min(k-1, p)
#
#    if dims == 1
#        MW = (sqrt.(π) .* M)*W
#        UDVᵀ = svd!(MW, full=false)
#        Cᵀ = view(UDVᵀ.Vt, 1:d, :)
#    else
#        WM = W*(transpose(sqrt.(π)) .* M)
#        UDVᵀ = svd!(WM, full=false)
#        Cᵀ = view(UDVᵀ.U, :, 1:d)
#    end
#
#    return transpose(Cᵀ)
#end


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
