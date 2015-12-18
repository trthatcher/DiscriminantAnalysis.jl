## Linear Discriminant Plots

function hyperplane{T<:AbstractFloat}(Ω::Matrix{T}, M::Matrix{T}, priors::Vector{T}, i, j)
    μ_i = vec(M[i,:])
    μ_j = vec(M[j,:])
    a = (μ_j - μ_i)'Ω
    c = sum(μ_i'Ω*μ_i - μ_j'Ω*μ_j)/2 + log(priors[i]/priors[j])
    (vec(a), c)
end

function hyperplane_x2{T<:AbstractFloat}(a::Vector{T}, c::T, x1::T)
    length(a) == 2 || throw(DimensionMismatch(""))
    x2 = -(a[1]*x1 + c)/a[2]
end

function hyperplane_x3{T<:AbstractFloat}(a::Vector{T}, c::T, x1::T, x2::T)
    length(a) == 3 || throw(DimensionMismatch(""))
    x3 = -(a[1]*x1 + a[2]*x2 + c)/a[3]
end

# Loading Data

using DataFrames

iris_df = readtable("iris.csv")
pool!(iris_df, [:Species])  # Ensure species is made a pooled data vector
y = iris_df[:Species].refs  # Class indices

using PyPlot, DiscriminantAnalysis
DA = DiscriminantAnalysis

X = convert(Array{Float64}, iris_df[[:PetalWidth, :SepalWidth]])
plt_x1 = vec(X[:,1])
plt_x2 = vec(X[:,2])
x2_max = maximum(plt_x2)
x2_min = minimum(plt_x2)

M = DA.class_means(X, y)
H = DA.center_classes!(copy(X), M, y)
Ω = inv(H'H/(size(X,1)-1))
π_k = Float64[1/3; 1/3; 1/3]

a_12, c_12 = hyperplane(Ω, M, π_k, 1, 2)
a_23, c_23 = hyperplane(Ω, M, π_k, 2, 3)

function δ_12{T<:AbstractFloat}(x1::T)
    x2 = hyperplane_x2(a_12, c_12, x1)
    (x2 > x2_max || x2 < x2_min) ? convert(T,NaN) : x2
end

function δ_23{T<:AbstractFloat}(x1::T)
    x2 = hyperplane_x2(a_23, c_23, x1)
    (x2 > x2_max || x2 < x2_min) ? convert(T,NaN) : x2
end


PyPlot.figure("Linear Discriminant Analysis")
PyPlot.scatter(plt_x1[y .== 1], plt_x2[y .== 1], s=40*ones(plt_x1[y .== 1]), c="r")
PyPlot.scatter(plt_x1[y .== 2], plt_x2[y .== 2], s=40*ones(plt_x1[y .== 2]), c="m")
PyPlot.scatter(plt_x1[y .== 3], plt_x2[y .== 3], s=40*ones(plt_x1[y .== 3]), c="b")

x = linspace(0,2.5,100); 
y = Float64[δ_12(x) for x in x]
plot(x, y, color="red", linewidth=2.0, linestyle="--")


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
