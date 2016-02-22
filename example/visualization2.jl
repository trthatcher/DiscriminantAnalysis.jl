using DiscriminantAnalysis, PyPlot

DA = DiscriminantAnalysis

function rotationmatrix2D{T<:AbstractFloat}(θ::T)
    T[cos(θ) -sin(θ);
      sin(θ)  cos(θ)]
end

function boxmuller(n::Integer)
    u1 = rand(n)
    u2 = rand(n)
    Z = Float64[(√(-2log(u1)) .* cos(2π*u2)) (√(-2log(u1)) .* sin(2π*u2))]
end


#==========================================================================
  Linear Discriminant  
==========================================================================#


# f(x) = bᵀx + c
function line{T<:AbstractFloat}(Ω::Matrix{T}, M::Matrix{T}, priors::Vector{T}, i, j)
    μ_i = vec(M[i,:])
    μ_j = vec(M[j,:])
    b = (μ_j - μ_i)'Ω
    c = sum(μ_i'Ω*μ_i - μ_j'Ω*μ_j)/2 + log(priors[i]/priors[j])
    (vec(b), c)
end

# Given x1, solve for x2 where 0 = bᵀx + c
function line_x2{T<:AbstractFloat}(b::Vector{T}, c::T, x1::T)
    length(b) == 2 || throw(DimensionMismatch(""))
    x2 = -(b[1]*x1 + c)/b[2]
end


#==========================================================================
  Quadratic Discriminant  
==========================================================================#

# f(x) = xᵀAx + bᵀx + c
function quadratic{T<:AbstractFloat}(Σ_k::Vector{Matrix{T}}, M::Matrix{T}, priors::Vector{T}, i ,j)
    μ_i = vec(M[i,:])
    μ_j = vec(M[j,:])
    Ω_i = inv(Σ_k[i])
    Ω_j = inv(Σ_k[j])
    A = (Ω_j - Ω_i)/2
    b = μ_i'Ω_i - μ_j'Ω_j
    c = sum(μ_j'Ω_j*μ_j - μ_i'Ω_i*μ_i)/2 + log(det(Ω_i)/det(Ω_j))/2 + log(priors[i]/priors[j])
    (A, vec(b), c)
end

# Given x1, solve for x2 where 0 = xᵀAx + bᵀx + c
function quadratic_x2{T<:AbstractFloat}(A::Matrix{T}, b::Vector{T}, c::T, x1::T)
    size(A) == (2,2) || throw(DimensionMismatch(""))
    length(v) == 2 || throw(DimensionMismatch(""))
    v1 = A[2,2]
    v2 = 2A[1,2]*x1 + b[2]
    v3 = A[1,1]*x1^2 + b[1]*x1 + c
    tmp = v2^2 - 4v1*v3
    tmp >= 0 ? (sqrt(tmp) - b)/(2a) : convert(T, NaN)
end

#θ = 2/atan((A[1,1] - A[2,2])/A[1,2])


n = 250

Z1 = boxmuller(n)
σ1 = [0.5 2]
X1 = ((Z1 .* σ1) * rotationmatrix2D(π/4)) .+ [-3 3]

Z2 = boxmuller(n)
σ2 = [1 2]
X2 = ((Z2 .* σ2) * rotationmatrix2D(-π/4)) .+ [3 3]

Z3 = boxmuller(n)
σ3 = [1.5 1.5]
X3 = (Z3 .* σ3) .+ [0 -1]

X = vcat(X1,X2,X3)
y = repeat([1,2,3], inner=[n])

M = DA.class_means(X, y)
H = DA.center_classes!(copy(X), M, y)
Σ_k = DA.class_covariances(H, y)
π_k = Float64[1/3; 1/3; 1/3]

A_12, b_12, c_12 = quadratic(Σ_k, M, π_k, 1, 2)
A_23, b_23, c_23 = quadratic(Σ_k, M, π_k, 2, 3)

#x1 = linspace(minimum(X[:,1]), maximum(X[:,1]), 300)
#PyPlot.scatter(X[y .== 1,1], X[y .== 1,2], c="r", zorder=3)
#PyPlot.scatter(X[y .== 2,1], X[y .== 2,2], c="b", zorder=4)
#PyPlot.scatter(X[y .== 3,1], X[y .== 3,2], c="m", zorder=5)

#plot(x1, Float64[quadratic_x2(A_12, b_12, c_12, x) for x in x1], color="r", linewidth=2.0, linestyle="--", zorder=1)

