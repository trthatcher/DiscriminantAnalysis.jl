struct LinearDiscriminantModel{T} <: DiscriminantModel{T}
    dims::Int
    "Whitening matrix for overall covariance matrix"
    W::AbstractMatrix{T}
    "Matrix of class means (one per row/column depending on `dims`)"
    M::Matrix{T}
    "Vector of class prior probabilities"
    π::Vector{T}
    "Subspace basis vectors reduced dimensions in canonical discriminant models"
    V::Union{Nothing, Matrix{T}}
    "Shrinkage parameter"
    γ::Union{Nothing, T}
end

function canonical_coordinates(M::Matrix, Σ⁻½::Matrix, π::Vector)
    Mc = M .- π'M



end

