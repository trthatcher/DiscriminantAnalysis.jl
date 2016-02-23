using DiscriminantAnalysis, Gadfly

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

#=
# Given x1, solve for x2 where 0 = xᵀAx + bᵀx + c
function rotate_quadratic{T<:AbstractFloat}(A::Matrix{T}, b::Vector{T})
    size(A) == (2,2) || throw(DimensionMismatch(""))
    length(v) == 2 || throw(DimensionMismatch(""))
    θ = atan(2A[3]/(A[1] - A[4]))
    R = rotationmatrix2D(θ)
    Ap = R * A * R'  # since A = R'DR
    bp = R * b
    (Ap, Bp, θ)
    #=
    v1 = A[2,2]
    v2 = 2A[1,2]*x1 + b[2]
    v3 = A[1,1]*x1^2 + b[1]*x1 + c
    tmp = v2^2 - 4v1*v3
    tmp >= 0 ? (sqrt(tmp) - b)/(2a) : convert(T, NaN)
    =#
end

=#

n = 250

Z1 = boxmuller(n)
σ1 = [0.5 3]
X1 = (Z1 .* σ1)

Z2 = boxmuller(n)
σ2 = [3.0 1.5]
X2 = (Z2 .* σ2) .+ [3 3]

X = vcat(X1,X2)
y = repeat([1,2], inner=[n])


xmin = minimum(X[:,1])
xmax = maximum(X[:,1])

ymin = minimum(X[:,2])
ymax = maximum(X[:,2])

M = DA.class_means(X, y)
H = DA.center_classes!(copy(X), M, y)
Σ_k = DA.class_covariances(H, y)
π_k = [0.5; 0.5]

A, b, c = quadratic(Σ_k, M, π_k, 1, 2)

#z(x,y) = ([x;y]'A*[x;y])[1] + (b*[x;y])[1] + c_

function eval_quad{T<:AbstractFloat}(A::Matrix{T}, b::Vector{T}, c::T, x::T, y::T)
    xy = x*y
    x² = x^2
    y² = y^2
    A[1]*x² + 2A[2]*xy + A[4]*y² + b[1]*x + b[2]*y + c
end

δ_23(x,y) = eval_quad(A, b, c, x, y)

plot(z = δ_23, x=linspace(xmin,xmax,150), y=linspace(ymin,ymax,150), Geom.contour(levels=[0.0]))




#x1 = linspace(minimum(X[:,1]), maximum(X[:,1]), 300)
#PyPlot.scatter(X[y .== 1,1], X[y .== 1,2], c="r", zorder=3)
#PyPlot.scatter(X[y .== 2,1], X[y .== 2,2], c="b", zorder=4)
#PyPlot.scatter(X[y .== 3,1], X[y .== 3,2], c="m", zorder=5)

#plot(x1, Float64[quadratic_x2(A_12, b_12, c_12, x) for x in x1], color="r", linewidth=2.0, linestyle="--", zorder=1)
