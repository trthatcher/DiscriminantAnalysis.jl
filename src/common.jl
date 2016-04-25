#==========================================================================
  Common Methods
==========================================================================#

#== Reference Vector ==#

immutable RefVector{T<:Integer} <: AbstractVector{T}
    ref::Vector{T}
    k::T
    function RefVector(ref::Vector{T}, k::T, check_integrity::Bool = true)
        if check_integrity
            if (refmin = minimum(ref)) <= 0
                error("Class reference should begin at 1; value $refmin found")
            end
            if (refmax = maximum(ref)) > k
                error("Class reference should not exceed $k; value $refmax found")
            end
            if length(unique(ref)) != k 
                error("A class between 1 and $k is not referenced.")
            end
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

for (scheme, dim_obs) in ((:(:row), 1), (:(:col), 2))
    isrowmajor = dim_obs == 1
    dim_param  = isrowmajor ? 2 : 1

    M_p = isrowmajor ? :p : :k
    M_k = isrowmajor ? :k : :p
    M_i = isrowmajor ? :(y[I[1]]) : :(I[1])
    M_j = isrowmajor ? :(I[2])    : :(y[I[2]])

    nk_k = isrowmajor ? :k : :1
    nk_1 = isrowmajor ? :1 : :k

    @eval begin
        function classtotals{T<:AbstractFloat}(::Type{Val{$scheme}}, X::Matrix{T}, y::RefVector)
            if length(y) != size(X, $dim_obs)
                errorstring = string("Dimension ", $dim_obs, " of X must match length of y")
                throw(DimensionMismatch(errorstring))
            end
            k = convert(Int64, y.k)
            p = size(X, $dim_param)
            M = zeros(T, $M_k, $M_p)
            for I in CartesianRange(size(X))
                M[$M_i,$M_j] += X[I]
            end
            M
        end

        function classmeans{T<:AbstractFloat}(::Type{Val{$scheme}}, X::Matrix{T}, y::RefVector)
            M  = classtotals(Val{$scheme}, X, y)
            nk = classcounts(y)
            k  = convert(Int64, y.k)
            broadcast!(/, M, M, reshape(nk, $nk_k, $nk_1))
        end

        function centerclasses!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}},
                X::Matrix{T}, 
                M::Matrix{T}, 
                y::RefVector
            )
            n, m = size(X)
            if !(size(M, $dim_param) == size(X, $dim_param))
                errorstring = string("X and M must match in dimension ", $dim_param)
                throw(DimensionMismatch(errorstring))
            end
            if !(y.k == size(M, $dim_obs))
                errorstring = string("Class count must match dimension ", $dim_obs, " of M")
                throw(DimensionMismatch(errorstring))
            end
            for I in CartesianRange(size(X))
                X[I] -= M[$M_i,$M_j]
            end
            X
        end

        function dotvectors!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}},
                X::AbstractMatrix{T}, 
                xᵀx::Vector{T}
            )
            if !(size(X,$dim_obs) == length(xᵀx))
                errorstring = string("Dimension mismatch on dimension ", $dim_obs)
                throw(DimensionMismatch(errorstring))
            end
            fill!(xᵀx, zero(T))
            for I in CartesianRange(size(X))
                xᵀx[I.I[$dim_obs]] += X[I]^2
            end
            xᵀx
        end
        function dotvectors{T<:AbstractFloat}(::Type{Val{$scheme}}, X::AbstractMatrix{T})
            dotvectors!(Val{$scheme}, X, Array(T, size(X, $dim_obs)))
        end
    end
end

subvector(::Type{Val{:row}}, M::Matrix, j::Integer) = sub(M, j, :)
subvector(::Type{Val{:col}}, M::Matrix, j::Integer) = sub(M, :, j)


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
#   Σ = VD²Vᵀ, WᵀΣW = I  =>  WᵀVD²VᵀW = (DVᵀW)ᵀ(DVᵀW)  =>  W = VD⁻¹
function whitendata_svd!{T<:BlasReal}(H::Matrix{T}, γ::T)
    0 <= γ <= 1 || error("Parameter γ=$(γ) must be in the interval [0,1]")
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

# Returns right-multiplied W for row-based observations; Z = XW
function whitendata_svd!{T<:BlasReal}(::Type{Val{:row}}, H::Matrix{T}, γ::T)
    transpose(whitendata_svd!(H, γ))
end

# Returns left-multiplied W for column-based observations; Z = WX
function whitendata_svd!{T<:BlasReal}(::Type{Val{:col}}, H::Matrix{T}, γ::T)
    whitendata_svd!(transpose(H), γ)
end


# Uses QR decomposition to whiten the implicit covariance matrix
#   Assumes H is row major, returns W
#   X = QR  =>  Σ = RᵀR, WᵀΣW = I  =>  W = R⁻¹
function whitendata_qr!{T<:BlasReal}(H::Matrix{T})
    n, m = size(H)
    if n <= m
        error("""Insufficient within-class observations to produce a full rank covariance matrix. 
                 Collect more data or consider regularization.""")
    end
    ϵ = eps(T) * n * m * maximum(H)
    QR = qrfact!(H, Val{false})
    R = triu!(QR[:R])
    if !all(abs(diag(R)) .>= ϵ)
        error("""Rank deficiency (collinearity) detected with tolerance $(ϵ). Ensure that all 
                 classes have sufficient observations to produce a full-rank covariance matrix.""")
    end
    W = LAPACK.trtri!('U', 'N', R)
    UpperTriangular(broadcast!(*, W, W, sqrt(n-one(T))))
end

# Returns right-multiplied W for row-based observations; Z = XW
whitendata_qr!{T<:BlasReal}(::Type{Val{:row}}, H::Matrix{T}) = whitendata_qr!(H)

# Returns left-multiplied W for column-based observations; Z = WX
function whitendata_qr!{T<:BlasReal}(::Type{Val{:col}}, H::Matrix{T}) 
    transpose!(whitendata_qr!(transpose(H)))
end


# Uses a Cholesky decomposition to whiten a covariance matrix. Regularization parameter γ shrinks 
# towards average eigenvalue
#   Returns W
#   Σ = UᵀU, WᵀΣW = I  =>  W = U⁻¹
function whitencov_chol!{T<:BlasReal}(Σ::Matrix{T}, γ::Nullable{T})
    ϵ = eps(T) * prod(size(Σ)) * maximum(Σ)
    if !isnull(γ) && get(γ) != 0
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
