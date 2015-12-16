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

using DiscriminantAnalysis
MOD = DiscriminantAnalysis

using Gadfly

X = convert(Array{Float64}, iris_df[[:PetalWidth, :SepalWidth]])

model = lda(X, y)
y_pred = classify(model, X)

M = MOD.class_means(X, y)
H = MOD.center_classes!(copy(X), M, y)
Ω = inv(H'H/(size(X,1)-1))
π_k = Float64[1/3; 1/3; 1/3]

a_12, c_12 = hyperplane(Ω, M, π_k, 1, 2)
a_13, c_13 = hyperplane(Ω, M, π_k, 1, 3)
a_23, c_23 = hyperplane(Ω, M, π_k, 2, 3)

δ_12(x1::AbstractFloat) = hyperplane_x2(a_12, c_12, x1)
δ_13(x1::AbstractFloat) = hyperplane_x2(a_13, c_13, x1)
δ_23(x1::AbstractFloat) = hyperplane_x2(a_23, c_23, x1)

err = vec(y .!= y_pred)

plt_x1 = vec(X[:,1])[err]
plt_x2 = vec(X[:,2])[err]
plt_y = y[err]

plt = plot(layer(x=plt_x1, y=plt_x2, color=plt_y, Geom.point),
           layer(δ_12, 0.0, 2.5),
           layer(δ_13, 0.0, 2.5),
           layer(δ_23, 0.0, 2.5))

draw(SVG("visualization_wrong.svg", 6inch, 6inch), plt)

#= 3d stuff

X = convert(Array{Float64}, iris_df[[:PetalWidth, :PetalLength, :SepalWidth]])
y = iris_df[:Species].refs  # Class indices

using DiscriminantAnalysis, PyPlot

MOD = DiscriminantAnalysis

PyPlot.figure("Iris Data")

#xlim(0.0,1.0)
#ylim(0.0,1.0)
#zlim(0.0,1.0)

for (k, colour) in ((1,"r"), (2,"m"), (3,"b"))
    class = y .== k
    PyPlot.scatter3D(vec(X[class,1]), vec(X[class,2]), vec(X[class,3]), c=colour, clip_on=true)
end

model = lda(X, y)

M = MOD.class_means(X, y)
H = MOD.center_classes!(X, M, y)
Σinv = inv(H'H/(size(X,1)-1))

function hyperplane3D(Σinv, μ_i, μ_j, x1, x2)  # aᵀx + c = 0
    a = (μ_j - μ_i)'Σinv
    c = (μ_i'Σinv*μ_i - μ_j'Σinv*μ_j)[1]/2
    x3 = -(a[1]*x1 + a[2]*x2 + c)/a[3]
end


x1 = Float64[0 4; 0 4]
x2 = Float64[8 8; 0 0]
x3 = reshape(Float64[hyperplane(Σinv, vec(M[2,:]), vec(M[3,:]), x1[i], x2[i]) for i = 1:length(x1)], 2, 2)

PyPlot.plot_surface(x1, x2, x3, 
                     rstride=1, cstride=1, alpha=0.25, clip_on = true)
=#
