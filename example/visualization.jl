#==========================================================================
  Helper Functions  
==========================================================================#

function hyperplane{T<:AbstractFloat}(Ω::Matrix{T}, M::Matrix{T}, priors::Vector{T}, i, j)
    μ_i = vec(M[i,:])
    μ_j = vec(M[j,:])
    v = (μ_j - μ_i)'Ω
    u = sum(μ_i'Ω*μ_i - μ_j'Ω*μ_j)/2 + log(priors[i]/priors[j])
    (vec(v), u)
end

function hyperplane_x2{T<:AbstractFloat}(a::Vector{T}, c::T, x1::T)
    length(a) == 2 || throw(DimensionMismatch(""))
    x2 = -(a[1]*x1 + c)/a[2]
end

function hyperplane_x3{T<:AbstractFloat}(a::Vector{T}, c::T, x1::T, x2::T)
    length(a) == 3 || throw(DimensionMismatch(""))
    x3 = -(a[1]*x1 + a[2]*x2 + c)/a[3]
end

# δ_i = δ_j -> δ_i - δ_j = 0
function quadratic{T<:AbstractFloat}(Σ_k::Vector{Matrix{T}}, M::Matrix{T}, priors::Vector{T}, i ,j)
    μ_i = vec(M[i,:])
    μ_j = vec(M[j,:])
    Ω_i = inv(Σ_k[i])
    Ω_j = inv(Σ_k[j])
    Q = (Ω_j - Ω_i)/2
    v = μ_i'Ω_i - μ_j'Ω_j
    u = sum(μ_j'Ω_j*μ_j - μ_i'Ω_i*μ_i)/2 + log(det(Ω_i)/det(Ω_j))/2 + log(priors[i]/priors[j])
    (Q, vec(v), u)
end

function quadratic_x2{T<:AbstractFloat}(Q::Matrix{T}, v::Vector{T}, u::T, x1::T)
    size(Q) == (2,2) || throw(DimensionMismatch(""))
    length(v) == 2 || throw(DimensionMismatch(""))
    a = Q[2,2]
    b = Q[1,2]*2x1 + v[2]
    c = Q[1,1]*x1^2 + v[1]*x1 + u
    tmp = b^2 - 4a*c
    tmp >= 0 ? (sqrt(tmp) - b)/(2a) : convert(T, NaN)
end


# Loading Data

using DataFrames

iris_df = readtable("iris.csv")
pool!(iris_df, [:Species])  # Ensure species is made a pooled data vector
y = iris_df[:Species].refs  # Class indices

using PyPlot, DiscriminantAnalysis
DA = DiscriminantAnalysis


#==========================================================================
  Linear Discriminant - Visualization
==========================================================================#

X = convert(Array{Float64}, iris_df[[:PetalWidth, :SepalWidth]])
plt_x1 = vec(X[:,1])
plt_x2 = vec(X[:,2])

M = DA.class_means(X, y)
H = DA.center_classes!(copy(X), M, y)
Ω = inv(H'H/(size(X,1)-1))
π_k = Float64[1/3; 1/3; 1/3]

a_12, c_12 = hyperplane(Ω, M, π_k, 1, 2)
a_23, c_23 = hyperplane(Ω, M, π_k, 2, 3)


PyPlot.figure("Linear Discriminant Analysis", figsize=(8,5))
PyPlot.xlabel("Petal Width")
PyPlot.ylabel("Sepal Width")

x_min = 0.0; x_max = 2.6; y_min = 1.85; y_max = 4.6;

PyPlot.ylim([y_min, y_max])
PyPlot.xlim([x_min, x_max])

function δ_12{T<:AbstractFloat}(x::T)
    y = hyperplane_x2(a_12, c_12, x)
    (y > y_max || y < y_min) ? convert(T,NaN) : y
end

function δ_23{T<:AbstractFloat}(x::T)
    y = hyperplane_x2(a_23, c_23, x)
    (y > y_max || y < y_min) ? convert(T,NaN) : y
end

x = linspace(x_min, x_max, 200); 
plot(x, Float64[δ_12(x) for x in x], color="r", linewidth=2.0, linestyle="--", zorder=1)
plot(x, Float64[δ_23(x) for x in x], color="b", linewidth=2.0, linestyle="--", zorder=2)

PyPlot.scatter(plt_x1[y .== 1], plt_x2[y .== 1], s=40*ones(plt_x1[y .== 1]), c="r", zorder=3)
PyPlot.scatter(plt_x1[y .== 2], plt_x2[y .== 2], s=40*ones(plt_x1[y .== 2]), c="m", zorder=4)
PyPlot.scatter(plt_x1[y .== 3], plt_x2[y .== 3], s=40*ones(plt_x1[y .== 3]), c="b", zorder=5)

PyPlot.text(0.3, 4.1, "Setosa", fontsize=20, zorder=6)
PyPlot.text(1.1, 3.5, "Versicolor", fontsize=20)
PyPlot.text(1.8, 2.1, "Virginica", fontsize=20)


#==========================================================================
  Canonical Discriminant - Visualization
==========================================================================#

X = convert(Array{Float64}, iris_df[[:PetalWidth, :SepalLength, :SepalWidth]])
plt_x1 = vec(X[:,1])
plt_x2 = vec(X[:,2])
plt_x3 = vec(X[:,3])

PyPlot.figure("Iris Scatter Plot", figsize=(8,6.5))

PyPlot.scatter3D(plt_x1[y .== 1], plt_x2[y .== 1], plt_x3[y .== 1], s=40*ones(plt_x1[y .== 1]), c="r")
PyPlot.scatter3D(plt_x1[y .== 2], plt_x2[y .== 2], plt_x3[y .== 2], s=40*ones(plt_x1[y .== 2]), c="m")
PyPlot.scatter3D(plt_x1[y .== 3], plt_x2[y .== 3], plt_x3[y .== 3], s=40*ones(plt_x1[y .== 3]), c="b")

Model = cda(X, y)

U = X * Model.W

plt_u1 = vec(U[:,1])
plt_u2 = vec(U[:,2])

PyPlot.figure("Canonical Discriminant Analysis", figsize=(8,5))

PyPlot.scatter(plt_u1[y .== 1], plt_u2[y .== 1], s=40*ones(plt_x1[y .== 1]), c="r", zorder=3)
PyPlot.scatter(plt_u1[y .== 2], plt_u2[y .== 2], s=40*ones(plt_x1[y .== 2]), c="m", zorder=4)
PyPlot.scatter(plt_u1[y .== 3], plt_u2[y .== 3], s=40*ones(plt_x1[y .== 3]), c="b", zorder=5)


#==========================================================================
  Quadratic Discriminant - Visualization
==========================================================================#

X = convert(Array{Float64}, iris_df[[:PetalWidth, :SepalWidth]])
plt_x1 = vec(X[:,1])
plt_x2 = vec(X[:,2])

M = DA.class_means(X, y)
H = DA.center_classes!(copy(X), M, y)
Σ_k = DA.class_covariances(H, y)
π_k = Float64[1/3; 1/3; 1/3]

Q_12, v_12, u_12 = quadratic(Σ_k, M, π_k, 1, 2)
Q_23, v_23, u_23 = quadratic(Σ_k, M, π_k, 2, 3)

PyPlot.figure("Quadratic Discriminant Analysis", figsize=(8,5))
PyPlot.xlabel("Petal Width")
PyPlot.ylabel("Sepal Width")

x_min = 0.0; x_max = 2.6; y_min = 1.85; y_max = 4.6;

PyPlot.ylim([y_min, y_max])
PyPlot.xlim([x_min, x_max])

function δ_12{T<:AbstractFloat}(x1::T)
    y = quadratic_x2(Q_12, v_12, u_12, x)
    y#(y > y_max || y < y_min) ? convert(T,NaN) : y
end

function δ_23{T<:AbstractFloat}(x1::T)
    y = quadratic_x2(Q_23, v_23, u_23, x)
    y#(y > y_max || y < y_min) ? convert(T,NaN) : y
end

x = linspace(x_min, x_max, 300);
plot(x, Float64[δ_12(x) for x in x], color="r", linewidth=2.0, linestyle="--", zorder=1)
plot(x, Float64[δ_23(x) for x in x], color="b", linewidth=2.0, linestyle="--", zorder=2)

PyPlot.scatter(plt_x1[y .== 1], plt_x2[y .== 1], s=40*ones(plt_x1[y .== 1]), c="r", zorder=3)
PyPlot.scatter(plt_x1[y .== 2], plt_x2[y .== 2], s=40*ones(plt_x1[y .== 2]), c="m", zorder=4)
PyPlot.scatter(plt_x1[y .== 3], plt_x2[y .== 3], s=40*ones(plt_x1[y .== 3]), c="b", zorder=5)

PyPlot.text(0.3, 4.1, "Setosa", fontsize=20, zorder=6)
PyPlot.text(1.1, 3.5, "Versicolor", fontsize=20)
PyPlot.text(1.8, 2.1, "Virginica", fontsize=20)




#=
using Gadfly
plt = plot(layer(x=plt_x1, y=plt_x2, color=plt_y, Geom.point),
           layer(f, minimum(plt_x1), maximum(plt_x1), color=2),
           #layer(δ_13, 0.0, 2.5),
           #layer(δ_23, 0.0, 2.5),
           Scale.color_discrete_manual(colorant"red",colorant"purple",colorant"blue"))#,
           #Scale.x_continuous(minvalue=0.0, maxvalue=1.0),
           #Scale.y_continuous(minvalue=0.0, maxvalue=1.0))

draw(SVG("visualization.svg", 6inch, 4inch), plt)
=#
