#==========================================================================
  Common Methods
==========================================================================#

# Element-wise translate
function translate!{T<:AbstractFloat}(A::Array{T}, b::T)
    @inbounds for i = 1:length(A)
        A[i] += b
    end
    A
end
translate!{T<:AbstractFloat}(b::T, A::Array{T}) = translate!(A, b)

# A := A .+ b'
function translate!{T<:AbstractFloat}(b::Vector{T}, A::Matrix{T})
    (n = size(A,1)) == length(b) || throw(DimensionMismatch("first dimension of A does not match length of b"))
    @inbounds for j = 1:size(A,2), i = 1:n
        A[i,j] += b[i]
    end
    A
end

# A := b .+ A
function translate!{T<:AbstractFloat}(A::Matrix{T}, b::Vector{T})
    (n = size(A,2)) == length(b) || throw(DimensionMismatch("second dimension of A does not match length of b"))
    @inbounds for j = 1:n, i = 1:size(A,1)
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

# S1 := (1-λ)S1+ λ*diagm(s2)
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


# Symmetrize the lower half of matrix S using the upper half of S
function symml!(S::Matrix)
    (p = size(S,1)) == size(S,2) || throw(ArgumentError("S must be square"))
    @inbounds for j = 1:(p - 1), i = (j + 1):p 
        S[i, j] = S[j, i]
    end
    S
end
symml(S::Matrix) = symml!(copy(S))

# sum(X .* X, 2)
function dot_columns{T<:AbstractFloat}(X::Matrix{T})
    n, p = size(X)
    xᵀx = zeros(p)
    for j = 1:p, i = 1:n
        xᵀx[j] += X[i,j]^2
    end
    xᵀx
end

# sum(X .* X, 1)
function dot_rows{T<:AbstractFloat}(X::Matrix{T})
    n, p = size(X)
    xᵀx = zeros(n)
    for j = 1:p, i = 1:n
        xᵀx[i] += X[i,j]^2
    end
    xᵀx
end

function class_counts{T<:Integer}(y::Vector{T}, k::T = maximum(y))
    counts = zeros(Int64, k)
    for i = 1:length(y)
        y[i] <= k || error("Index $i out of range.")
        counts[y[i]] += 1
    end
    counts
end

function class_totals{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, k::U = maximum(y))
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X and y must have the same number of rows."))
    M = zeros(T, k, p)
    for j = 1:p, i = 1:n
        M[y[i],j] += X[i,j]
    end
    M
end

# Compute matrix of class means
#   X is uncentered data matrix
#   y is one-based vector of class IDs
function class_means{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, k = maximum(y))
    M = class_totals(X, y, k)
    n_k = class_counts(y, k)
    scale!(one(T) ./ n_k, M)
end

# Center rows of X based on class mean in M
#   X is uncentered data matrix
#   M is matrix of class means (one per row)
function center_rows!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::Vector{U})
    n, p = size(X)
    for j = 1:p, i = 1:n
        X[i,j] -= M[y[i],j]
    end
    X
end
