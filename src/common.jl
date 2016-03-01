#==========================================================================
  Common Methods
==========================================================================#

# Helper type to
immutable RefVector{T<:Integer}
    y::Vector{T}
    k::T
    function RefVector(y::Vector{T}, k::T)
        (ymin = minimum(y)) >  0 || error("Class reference should begin at 1; value $ymin found")
        (ymax = maximum(y)) <= k || error("Class reference should not exceed $k; value $ymax found")
        length(unique(y)) == k || error("A class between 1 and $k is not referenced.")
        new(y, k)
    end
end
RefVector{T<:Integer}(y::Vector{T}, k::T = maximum(y)) = RefVector{T}(y, k)

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

# if S = UΛUᵀ then Λ := (1-γ)Λ .+ γ*λ_avg  => S := (1-γ)S + γ*λ_avg
function regularize!{T<:AbstractFloat}(Λ::Vector{T}, γ::T)
    0 <= γ <= 1 || error("γ = $(γ) must be in the interval [0,1]")
    λ_avg = mean(Λ)
    @inbounds for i in eachindex(Λ)
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
    @inbounds for j = 1:(p - 1), i = (j + 1):p 
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
function class_means{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, k::U = maximum(y))
    M = class_totals(X, y, k)
    n_k = class_counts(y, k)
    scale!(one(T) ./ n_k, M)
end

# Center rows of X based on class mean in M
#   X is uncentered data matrix
#   M is matrix of class means (one per row)
function center_classes!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::Vector{U})
    @inbounds for ij in CartesianRange(size(X))
        X[ij] -= M[y[ij.I[1]],ij.I[2]]
    end
    X
end

# Compute the symmetric matrix
function gramian{T<:BlasReal}(H::Matrix{T}, α::T, symmetrize::Bool=true)
    p = size(H,2)
    Σ = BLAS.syrk!('U', 'T', α, H, zero(T), Array(T,p,p))
    symmetrize ? symml!(Σ) : Σ
end


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
    all(D .>= ϵ) || error("Rank deficiency (collinearity) detected with tolerance $(ϵ).")
    transpose(scale!(1 ./ D, Vᵀ))
end

# Uses an eigendecomposition to whiten a covariance matrix
# Regularization parameter γ shrinks towards average eigenvalue
function whiten_cov!{T<:BlasReal}(Σ::Matrix{T}, γ::Nullable{T})
    ϵ = eps(T) * prod(size(Σ)) * maximum(Σ)
    Λ, V = LAPACK.syev!('V', 'U', Σ)
    if !isnull(γ)
        regularize!(Λ, get(γ))
    end
    all(Λ .>= ϵ) || error("Rank deficiency (collinearity) detected with tolerance $(ϵ).")
    scale!(V, 1 ./ sqrt(Λ))
end
