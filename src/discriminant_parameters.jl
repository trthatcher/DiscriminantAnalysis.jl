### Model Parameters

mutable struct DiscriminantParameters{T}
    "Model fit indicator - `true` if model has been fit"
    fit::Bool
    "Dimension along which observations are stored (1 for rows, 2 for columns)"
    dims::Int
    "Number of classes"
    m::Int
    "Number of predictors"
    p::Int
    "Matrix of class centroids (one per row or column - see `dims`)"
    M::Matrix{T}
    "Prior-weighted overall centroid"
    μ::Vector{T}
    "Vector of class prior probabilities"
    π::Vector{T}
    "Vector of Class counts"
    nₘ::Vector{Int}
    "Shrinkage parameter"
    γ::Union{Nothing,T}
    "Overall covariance matrix without `γ` regularization"
    Σ::Union{Nothing,Matrix{T}}
    function DiscriminantParameters{T}() where T
        new{T}(false, 0, 0, 0, Matrix{T}(undef,0,0), Vector{T}(undef,0), Vector{T}(undef,0),
               Vector{Int}(undef,0), nothing, nothing)
    end
end


function set_dimensions!(Θ::DiscriminantParameters{T},
                         y::Vector{<:Integer},
                         X::Matrix{T},
                         dims::Integer) where {T}
    n, p = check_dims(X, dims=dims)
    m = maximum(y)
    
    n₂ = length(y)
    n₂ == n || throw(DimensionMismatch("observation count along length of class index " *
                                       "vector y must match dimension $(dims) of data " *
                                       "matrix X (got $(n₂) and $(n))"))

    Θ.dims = dims
    Θ.m = m
    Θ.p = p

    return Θ
end


function set_gamma!(Θ::DiscriminantParameters{T}, γ::Union{Nothing,Real}) where T
    if γ !== nothing
        0 ≤ γ ≤ 1 || throw(DomainError(γ, "γ must be in the interval [0,1]"))
    end
    Θ.γ = γ

    return Θ
end


function set_statistics!(Θ::DiscriminantParameters{T},
                         y::Vector{<:Integer},
                         X::Matrix{T},
                         centroids::Union{Nothing,AbstractMatrix}) where T
    dims = Θ.dims
    m = Θ.m
    p = Θ.p

    # Compute centroids and class counts from data if not specified
    if centroids === nothing
        Θ.M = dims == 1 ? Matrix{T}(undef, m, p) : Matrix{T}(undef, p, m)
        Θ.nₘ = Vector{Int}(undef, m)

        class_statistics!(Θ.M, Θ.nₘ, X, y, dims=dims)
    else
        m₂, p₂ = check_dims(centroids, dims=dims)
        if m₂ != m
            throw(DimensionMismatch("class count along dimension $(dims) of centroid " * 
                                    "matrix M must match maximum class index found in " *
                                    "class index vector y (got $(m₂) and $(m))"))
        elseif p₂ != p
            altdims = dims == 1 ? 2 : 1
            throw(DimensionMismatch("predictor count along dimension $(altdims) of " *
                                    "centroid matrix M must match dimension $(dims) of " *
                                    "data matrix X (got $(p₂) and $(p))"))
        end

        Θ.M = copyto!(similar(centroids, T), centroids)
        Θ.nₘ = class_counts!(Vector{Int}(undef, m), y)
    end

    validate_class_counts(Θ.nₘ)

    return Θ
end


function set_priors!(Θ::DiscriminantParameters{T}, 
                     π::Union{Nothing,AbstractVector}) where T
    m = Θ.m

    # Compute priors from class frequencies in data if not specified
    if π === nothing
        Θ.π = broadcast!(/, Vector{T}(undef, m), Θ.nₘ, sum(Θ.nₘ))
    else
        m₂ = length(π)
        if m₂ != m
            throw(DimensionMismatch("class count along length of class prior probability " *
                                    "vector π must match maximum class index found in " *
                                    "class index vector y (got $(m₂) and $(m))"))
        end
        validate_priors(π)

        Θ.π = copyto!(similar(π, T), π)
    end

    return Θ
end