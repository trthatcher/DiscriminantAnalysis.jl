function discriminants(LDA::LinearDiscriminantModel{T}, X::Matrix{T}) where T
    dims = LDA.Θ.dims
    n = size(X, dims)
    m = LDA.Θ.m

    Δ = dims == 1 ? Matrix{T}(undef, n, m) : Matrix{T}(undef, m, n)

    return discriminants!(Δ, LDA, X)
end


function _posteriors!(Δ::Matrix{T}, LDA::LinearDiscriminantModel{T}) where T
    Π = broadcast!(exp, Δ, Δ)
    alt_dims = LDA.Θ.dims == 1 ? 2 : 1
    return broadcast!(/, Π, Π, sum(Π, dims=alt_dims))
end


function posteriors!(Π::Matrix{T}, LDA::LinearDiscriminantModel{T}, X::Matrix{T}) where T
    Δ = discriminants!(Π, LDA, X)
    return _posteriors!(Δ, LDA)
end


function posteriors(LDA::LinearDiscriminantModel{T}, X::Matrix{T}) where T
    Δ = discriminants(LDA, X)
    return _posteriors!(Δ, LDA)
end