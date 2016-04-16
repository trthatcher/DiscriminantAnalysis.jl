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

for (scheme, dimension) in ((:(:row), 1), (:(:col), 2))
    M_ref  = dimension == 1 ? :(M[y[I[1]],I[2]])          : :(M[I[1],y[I[2]]])
    M_init = dimension == 1 ? :(zeros(T, y.k, size(X,2))) : :(zeros(T, size(X,1), y.k))
    w_k    = dimension == 1 ? :(reshape(n_k, k, 1))       : :(reshape(n_k, 1, k))
    altdimension = dimension == 1 ? 2 : 1
    @eval begin
        function classtotals{T<:AbstractFloat}(::Type{Val{$scheme}}, X::Matrix{T}, y::RefVector)
            if length(y) != size(X, $dimension)
                errorstring = string("Dimension ", $dimension, " of X must match length of y")
                throw(DimensionMismatch(errorstring))
            end
            M = $M_init
            for I in CartesianRange(size(X))
                $M_ref += X[I]
            end
            M
        end

        function classmeans{T<:AbstractFloat}(::Type{Val{$scheme}}, X::Matrix{T}, y::RefVector)
            M   = classtotals(Val{$scheme}, X, y)
            n_k = classcounts(y)
            k = convert(Int64, y.k)
            broadcast!(/, M, M, $w_k)
        end

        function centerclasses!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}},
                X::Matrix{T}, 
                M::Matrix{T}, 
                y::RefVector
            )
            n, m = size(X)
            if !(size(M, $altdimension) == size(X, $altdimension))
                errorstring = string("X and M must match in dimension ", $altdimension)
                throw(DimensionMismatch(errorstring))
            end
            if !(y.k == size(M, $dimension))
                errorstring = string("Class count must match dimension ", $dimension, " of M")
                throw(DimensionMismatch(errorstring))
            end
            for I in CartesianRange(size(X))
                X[I] -= $M_ref
            end
            X
        end

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
        function dotvectors{T<:AbstractFloat}(::Type{Val{$scheme}}, X::AbstractMatrix{T})
            dotvectors!(Val{$scheme}, X, Array(T, size(X,$dimension)))
        end
    end
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


#== Whitening Functions ==#

# Random Vector: Cov(x) = E(xxᵀ) = Σ  =>  Cov(Wᵀx) = WᵀCov(x)W = WᵀΣW
# Row Major:     Cov(X) = XᵀX => Cov(XW) = WᵀXᵀXW
# Column Major:  Cov(X) = XXᵀ => Cov(WX) = WXXᵀWᵀ

# Uses SVD decomposition to whiten the implicit γ-regularized covariance matrix
#   Assumes H is row major, returns Wᵀ
#   Σ = VD²Vᵀ
#   WᵀΣW = I  =>  WᵀVD²VᵀW = (DVᵀW)ᵀ(DVᵀW)  =>  W = VD⁻¹
function whitendata_svd!{T<:BlasReal}(H::Matrix{T}, γ::T)
    n, m = size(H)
    ϵ = eps(T) * n * m * maximum(H)
    UDVᵀ = svdfact!(H)
    D = UDVᵀ.S
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
