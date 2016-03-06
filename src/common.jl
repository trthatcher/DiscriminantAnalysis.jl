#==========================================================================
  Common Methods
==========================================================================#

# Helper type for references
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

function class_counts{T<:Integer}(y::RefVector{T})
    counts = zeros(Int64, y.k)
    for i in eachindex(y)
        counts[y[i]] += 1
    end
    counts
end

function class_totals{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::RefVector{U})
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X and y must have the same number of rows."))
    M = zeros(T, y.k, p)
    for j = 1:p, i = 1:n
        M[y[i],j] += X[i,j]
    end
    M
end

# Compute matrix of class means
#   X is uncentered data matrix
#   y is one-based vector of class IDs
function class_means{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::RefVector{U})
    M   = class_totals(X, y)
    n_k = class_counts(y)
    scale!(one(T) ./ n_k, M)
end

# Center rows of X based on class mean in M
#   X is uncentered data matrix
#   M is matrix of class means (one per row)
function center_classes!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::RefVector{U})
    n, p = size(X)
    size(M,2) == p   || error("X and M must have the same number of columns.")
    size(M,1) == y.k || error("M should have as many rows as y has classes.")
    for j = 1:p, i = 1:n
        X[i,j] -= M[y[i],j]
    end
    X
end

# Element-wise translate
function translate!{T<:AbstractFloat}(A::Array{T}, b::T)
    for i = 1:length(A)
        A[i] += b
    end
    A
end
translate!{T<:AbstractFloat}(b::T, A::Array{T}) = translate!(A, b)

# A := A .+ b'
function translate!{T<:AbstractFloat}(b::Vector{T}, A::Matrix{T})
    (n = size(A,1)) == length(b) || throw(DimensionMismatch("first dimension of A does not match length of b"))
    for j = 1:size(A,2), i = 1:n
        A[i,j] += b[i]
    end
    A
end

# A := b .+ A
function translate!{T<:AbstractFloat}(A::Matrix{T}, b::Vector{T})
    (n = size(A,2)) == length(b) || throw(DimensionMismatch("second dimension of A does not match length of b"))
    for j = 1:n, i = 1:size(A,1)
        A[i,j] += b[j]
    end
    A
end

# S1 := (1-λ)S1+ λS2
function regularize!{T<:AbstractFloat}(S1::Matrix{T}, λ::T, S2::Matrix{T})
    (n = size(S1,1)) == size(S1,2) || throw(DimensionMismatch("Matrix S1 must be square."))
    (m = size(S2,1)) == size(S2,2) || throw(DimensionMismatch("Matrix S2 must be square."))
    n == m || throw(DimensionMismatch("Matrix S1 and S2 must be of the same order."))
    0 <= λ <= 1 || error("λ = $(λ) must be in the interval [0,1]")
    for j = 1:n, i = 1:n
            S1[i,j] = (1-λ)*S1[i,j] + λ*S2[i,j]
    end
    S1
end

# if S = UΛUᵀ then Λ := (1-γ)Λ .+ γ*λ_avg  => S := (1-γ)S + γ*λ_avg
function regularize!{T<:AbstractFloat}(Λ::Vector{T}, γ::T)
    0 <= γ <= 1 || error("γ = $(γ) must be in the interval [0,1]")
    λ_avg = mean(Λ)
    for i in eachindex(Λ)
        Λ[i] = (1-γ)*Λ[i] + γ*λ_avg
    end
    Λ
end


# S1 := (1-λ)S1+ λ*diagm(s2)
#=
function regularize!{T<:AbstractFloat}(S::Matrix{T}, λ::T, s::Vector{T})
    (n = size(S,1)) == size(S,2) || throw(DimensionMismatch("Matrix S must be square."))
    n == length(s) || throw(DimensionMismatch("The length vector s must be the order of S."))
    0 <= λ <= 1 || error("λ = $(λ) must be in the interval [0,1]")
    scale!(S, (1-λ))
    for i = 1:n
        S[i,i] += λ*s[i]
    end
    S
end
=#


# Symmetrize the lower half of matrix S using the upper half of S
function symml!(S::Matrix)
    (p = size(S,1)) == size(S,2) || throw(ArgumentError("S must be square"))
    for j = 1:(p - 1), i = (j + 1):p 
        S[i, j] = S[j, i]
    end
    S
end
symml(S::Matrix) = symml!(copy(S))

# sum(X .* X, 2)
#=
function dot_columns{T<:AbstractFloat}(X::Matrix{T})
    n, p = size(X)
    xᵀx = zeros(p)
    for j = 1:p, i = 1:n
        xᵀx[j] += X[i,j]^2
    end
    xᵀx
end
=#

function zero!{T<:Number}(A::AbstractArray{T})
    @inbounds for i in eachindex(A)
        A[i] = zero(T)
    end
    A
end

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
