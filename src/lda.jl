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
    function LinearDiscriminantModel{T}(M::Matrix{T}, π::Vector{T}, dims::Integer=1, 
                                        γ::Union{Nothing,Real}=nothing,
                                        canonical::Bool=false) where T
        dims ∈ (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))

        k = size(M, dims)
        p = size(M, dims == 1 ? 2 : 1)
        kₚ = length(π)

        k == kₚ || error("the length of class priors π should match the number of " *
                         (dims == 1 ? "rows" : "columns") * " of M (got $(kₚ) and $(k))")

        if isa(γ, Real)
            0 ≤ γ ≤ 1 || error("γ must be in the interval [0,1] (got $(γ))")
        end

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
    dims ∈ (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
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
function _fit!(LDA::LinearDiscriminantModel{T}, y::Vector{<:Integer}, X::Matrix{T}) where T
    _center_classes!(X, y, LDA.M, LDA.dims)

    if isa(LDA.γ, Real) 
        copyto!(LDA.W, _whiten_data!(X, LDA.γ, LDA.dims))
    else
        copyto!(LDA.W, _whiten_data!(X, LDA.dims))
    end
    
end