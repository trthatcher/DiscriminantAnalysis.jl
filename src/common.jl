#==========================================================================
  Common Methods
==========================================================================#

#== Reference Vector ==#

immutable RefVector{T<:Integer} <: AbstractVector{T}
    ref::Vector{T}
    k::T
    function RefVector(ref::Vector{T}, k::T, check_integrity::Bool = true)
        if check_integrity
            (refmin = minimum(ref)) >  0 || error("Class reference should begin at 1; value $refmin found")
            (refmax = maximum(ref)) <= k || error("Class reference should not exceed $k; value $refmax found")
            length(unique(ref)) == k || error("A class between 1 and $k is not referenced.")
        end
        new(copy(ref), k)
    end
end
function RefVector{T<:Integer}(y::Vector{T}, k::T = maximum(y), check_integrity::Bool = true)
    RefVector{T}(y, k, check_integrity)
end

Base.size(y::RefVector) = (length(y.ref),)
Base.linearindexing(::Type{RefVector}) = Base.LinearFast()
Base.getindex(y::RefVector, i::Int) = getindex(y.ref, i)

function convert{U<:Integer}(::Type{RefVector{U}}, y::RefVector)
    RefVector(copy(convert(Vector{U}, y.ref)), convert(U, y.k), false)
end


#== Class Matrix Functions ==#

function classcounts{T<:Integer}(y::RefVector{T})
    counts = zeros(Int64, y.k)
    for i in eachindex(y)
        counts[y[i]] += 1
    end
    counts
end

function classtotals{T<:AbstractFloat}(::Type{Val{:row}}, X::Matrix{T}, y::RefVector)
    n, p = size(X)
    if length(y) != n
        throw(DimensionMismatch("X and y must have the same number of rows."))
    end
    M = zeros(T, y.k, p)
    for j = 1:p, i = 1:n
        M[y[i],j] += X[i,j]
    end
    M
end

function classtotals{T<:AbstractFloat}(::Type{Val{:col}}, X::Matrix{T}, y::RefVector)
    p, n = size(X)
    if length(y) != n
        throw(DimensionMismatch("X and y must have the same number of columns."))
    end
    M = zeros(T, p, y.k)
    for j = 1:n, i = 1:p
        M[i,y[j]] += X[i,j]
    end
    M
end

# Compute matrix of class means
#   X is uncentered data matrix
#   y is one-based vector of class IDs
function classmeans{T<:AbstractFloat}(::Type{Val{:row}}, X::Matrix{T}, y::RefVector)
    M   = classtotals(Val{:row}, X, y)
    n_k = classcounts(y)
    scale!(one(T) ./ n_k, M)
end

function classmeans{T<:AbstractFloat}(::Type{Val{:col}}, X::Matrix{T}, y::RefVector)
    M   = classtotals(Val{:col}, X, y)
    n_k = classcounts(y)
    scale!(M, one(T) ./ n_k)
end


# Center rows of X based on class mean in M
#   X is uncentered data matrix
#   M is matrix of class means (one per row)
function centerclasses!{T<:AbstractFloat}(
         ::Type{Val{:row}},
        X::Matrix{T}, 
        M::Matrix{T}, 
        y::RefVector
    )
    n, p = size(X)
    size(M,2) == p   || throw(DimensionMismatch("X and M must have the same number of columns."))
    size(M,1) == y.k || throw(DimensionMismatch("M should have as many rows as y has classes."))
    for j = 1:p, i = 1:n
        X[i,j] -= M[y[i],j]
    end
    X
end

function centerclasses!{T<:AbstractFloat}(
         ::Type{Val{:col}},
        X::Matrix{T}, 
        M::Matrix{T}, 
        y::RefVector
    )
    p, n = size(X)
    size(M,2) == y.k || throw(DimensionMismatch("M should have as many columns as y has classes."))
    size(M,1) == p   || throw(DimensionMismatch("X and M must have the same number of rows."))
    for j = 1:p, i = 1:n
        X[i,j] -= M[y[i],j]
    end
    X
end


#== Regularization Functions ==#

# S1 := (1-λ)S1+ λS2
function regularize!{T<:AbstractFloat}(S1::Matrix{T}, λ::T, S2::Matrix{T})
    (n = size(S1,1)) == size(S1,2) || throw(DimensionMismatch("Matrix S1 must be square."))
    (m = size(S2,1)) == size(S2,2) || throw(DimensionMismatch("Matrix S2 must be square."))
    n == m || throw(DimensionMismatch("Matrix S1 and S2 must be of the same order."))
    0 <= λ <= 1 || error("λ = $(λ) must be in the interval [0,1]")
    for I in CartesianRange((n,n))
        S1[I] = (1-λ)*S1[I] + λ*S2[I]
    end
    S1
end

# S := (1-γ)S + γ*I*λ_avg
function regularize!{T<:AbstractFloat}(S::Matrix{T}, γ::T)
    (n = size(S,1)) == size(S,2) || throw(DimensionMismatch("Matrix S must be square."))
    0 <= γ <= 1 || error("γ = $(γ) must be in the interval [0,1]")
    λ_avg = trace(S)/n
    scale!(S, 1-γ)
    for i = 1:n
        S[i,i] += γ*λ_avg
    end
    S
end

for (scheme, dimension) in ((:row, 1), (:col, 2))
    function dotvectors!{T<:AbstractFloat}(
             ::Type{Val{$scheme}},
            X::AbstractMatrix{T}, 
            xᵀx::Vector{T}
        )
        if !(size(X,$dimension) == length(xᵀx))
            errorstring = string("Dimension mismatch on dimension ", $dimension)
            throw(DimensionMismatch(errorstring))
        end
        fill!(xᵀx, zero(T))
        for I in CartesianRange(size(X))
            xᵀx[I.I[$dimension]] += X[I]^2
        end
        xᵀx
    end
end


#== Whitening Functions ==#

#=
Random Vector: (assume centered)
    Cov(x) = E(xxᵀ) = Σ  =>  Cov(Wᵀx) = WᵀCov(x)W = WᵀΣW
    Σ = VD²Vᵀ
    WᵀΣW = I  =>  WᵀVD²VᵀW = (DVᵀW)ᵀ(DVᵀW)

Row-Major Data Matrix:
    Cov(X) = XᵀX => Cov(XW) = WᵀXᵀXW

Column-Major Data Matrix:
    Cov(X) = XXᵀ => Cov(WX

=#

# Uses SVD decomposition to whiten the implicit γ-regularized covariance matrix
# Assumes H is row major, returns Wᵀ 
function whitendata_svd!{T<:BlasReal}(H::Matrix{T}, γ::T)
    n, m = size(H)
    ϵ = eps(T) * n * m * maximum(H)
    UDVᵀ = svdfact!(H)
    D = UDVᵀ.D
    λ_avg = zero(T)
    @inbounds for i in eachindex(D)
        D[i] = (D[i]^2)/(n-1)  # λi for Σ = H'H/(n-1)
        λ_avg += D[i]
    end
    λ_avg /= length(D)
    @inbounds for i in eachindex(D)
        D[i] = sqrt((1-γ)*D[i] + γ*λ_avg)
    end
    if !all(D .>= ϵ)
        error("""Rank deficiency (collinearity) detected with tolerance $(ϵ). Ensure that all 
                 classes have sufficient observations to produce a full-rank covariance matrix.""")
    end
    broadcast!(/, UDVᵀ.Vt, UDVᵀ.Vt, D)
end

# Uses QR decomposition to whiten the implicit covariance matrix
# Assumes H is row major, returns W
function whitendata_qr!{T<:BlasReal}(H::Matrix{T})
    n, m = size(H)
    if n <= m
        error("""Insufficient observations to produce a full rank covariance matrix. Collect more
                 data or consider regularization.""")
    end
    ϵ = eps(T) * n * m * maximum(H)
    QR = qrfact!(H, pivot=Val{True})
    R = QR[:R]
    if !all(diag(R) .>= ϵ)
        error("""Rank deficiency (collinearity) detected with tolerance $(ϵ). Ensure that all 
                 classes have sufficient observations to produce a full-rank covariance matrix.""")
    end
    W = LAPACK.trtri!('U', 'N', R)
    UpperTriangular(broadcast!(/, W, W, sqrt(n-one(T))))
end

# Uses a Cholesky decomposition to whiten a covariance matrix
# Regularization parameter γ shrinks towards average eigenvalue
function whitencov_chol!{T<:BlasReal}(Σ::Matrix{T}, γ::Nullable{T})
    ϵ = eps(T) * prod(size(Σ)) * maximum(Σ)
    if !isnull(γ)
        regularize!(Σ, get(γ))
    end
    if !all(diag(Σ) .>= ϵ)
        error("""Rank deficiency (collinearity) detected with tolerance $(ϵ). Ensure that all 
                 classes have sufficient observations to produce a full-rank covariance matrix.""")
    end
    UᵀU = cholfact(Σ, :U, Val{false})
    U = triu!(UᵀU.factors)
    UpperTriangular(LAPACK.trtri!('U', 'N', U))
end






























#=

# sum(X .* X, 1)
function dotrows!{T<:AbstractFloat}(X::Matrix{T}, xᵀx::Vector{T})
    size(X,1) == length(xᵀx) || error("Vector xᵀx should have same number of rows as X")
    zero!(xᵀx)
    for I in CartesianRange(size(X))
        xᵀx[I.I[1]] += X[I]^2
    end
    xᵀx
end
dotrows{T<:AbstractFloat}(X::Matrix{T}) = dotrows!(X, Array(T, size(X,1)))

# Compute the symmetric matrix
gramian{T<:BlasReal}(H::Matrix{T}, α::T) = scale!(H'H, α)

# Uses a singular value decomposition to whiten a centered matrix H
# Regularization parameter γ shrinks towards average eigenvalue
function whiten_data!{T<:BlasReal}(H::Matrix{T}, γ::Nullable{T})
    n = size(H,1)
    ϵ = eps(T) * prod(size(H)) * maximum(H)
    _U, D, Vᵀ = LAPACK.gesdd!('A', H)  # Recall: Sw = H'H/(n-1) = VD²Vᵀ
    if !isnull(γ)
        Λ = regularize!(D.^2/(n-1), get(γ))
        @inbounds for i in eachindex(D)
            D[i] = sqrt(Λ[i])
        end
    else
        @inbounds for i in eachindex(D)
            D[i] /= sqrt(n-1)
        end
    end
    if !all(D .>= ϵ)
        error("""Rank deficiency (collinearity) detected with tolerance $(ϵ). Ensure that all 
                 classes have sufficient observations to produce a full-rank covariance matrix.""")
    end
    transpose(scale!(1 ./ D, Vᵀ))
end

# Uses an eigendecomposition to whiten a covariance matrix
# Regularization parameter γ shrinks towards average eigenvalue
function whiten_cov!{T<:BlasReal}(Σ::Matrix{T}, γ::Nullable{T})
    ϵ = eps(T) * prod(size(Σ)) * maximum(Σ)
    if ϵ > 1
        println(Σ)
        println(γ)
    end
    Λ, V = LAPACK.syev!('V', 'U', Σ)
    if !isnull(γ)
        regularize!(Λ, get(γ))
    end
    all(Λ .>= ϵ) || error("Rank deficiency (collinearity) detected with tolerance $(ϵ).")
    scale!(V, 1 ./ sqrt(Λ))
end
=#
