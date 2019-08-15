mutable struct LinearDiscriminantModel{T} <: DiscriminantModel{T}
    "whether the model has been fit"
    fit::Bool
    "Dimensions"
    dims::Int
    "Whitening matrix for overall covariance matrix"
    W::AbstractMatrix{T}
    "Matrix of class means (one per row/column depending on `dims`)"
    M::Matrix{T}
    "Vector of class prior probabilities"
    π::Vector{T}
    "Subspace basis vectors reduced dimensions in canonical discriminant models"
    C::Union{Nothing,Matrix{T}}
    "Shrinkage parameter"
    γ::Union{Nothing,T}
    function LinearDiscriminantModel{T}(M::AbstractMatrix, π::AbstractVector, 
                                        dims::Integer=1, γ::Union{Nothing,Real}=nothing,
                                        canonical::Bool=false) where T
        dims ∈ (1, 2) || arg_error("dims should be 1 or 2 (got $dims)")

        k = size(M, dims)
        p = size(M, dims == 1 ? 2 : 1)
        kₚ = length(π)

        k == kₚ || dim_error("the length of class priors π should match the number of " *
                             (dims == 1 ? "rows" : "columns") * " of centroid matrix M " *
                             "(got $(kₚ) and $(k))")

        if γ !== Nothing
            0 ≤ γ ≤ 1 || dom_error(γ, "γ must be in the interval [0,1]")
        end

        total = sum(π)
        isapprox(total, one(T)) || arg_error("class priors π must sum to 1 (got $(total))")
        all(π .≥ 0) || arg_error("all class priors πₖ must be non-negative probabilities")

        W = zeros(T, p, p)

        if canonical
            d = min(k-1, p)
            C = dims == 1 ? zeros(T, p, d) : zeros(T, d, p)
        else
            C = nothing
        end

        new{T}(false, dims, W, M, π, C, γ)
    end
end

function canonical_coordinates(M::Matrix{T}, W::Matrix{T}, dims::Integer) where T
    dims ∈ (1, 2) || arg_error("dims should be 1 or 2 (got $dims)")
    k = size(M, dims)
    p = size(M, dims == 1 ? 2 : 1)

    all(size(W) .== p) || dim_error("W must match dimensions of M")

    d = min(k-1, p)

    if dims == 1
        UDVᵀ = svd!(M*W, full=false)
        Cᵀ = view(UDVᵀ.Vt, 1:d, :)
    else
        UDVᵀ = svd!(W*M, full=false)
        Cᵀ = view(UDVᵀ.U, :, 1:d)
    end

    return transpose(Cᵀ)
end

"""
X is not centered, M is known
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
