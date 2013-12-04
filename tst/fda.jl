include("../src/DA.jl")

using DA

using DataFrames
using RDatasets



iris = data("datasets", "iris")

clean_colnames!(iris)

pool!(iris, ["Species"])

fm = Formula(:(Species ~ Sepal_Length + Sepal_Width + Petal_Length + Petal_Width))

y = iris[["Species"]][1]

mf = ModelFrame(fm, iris)
X = ModelMatrix(mf).m[:,2:]
n,p = size(X)

M = zeros(Float64,3,p)
for i=1:n
	M[y.refs[i],:] += X[i,:]
end

println("Good1")

M = M ./ vec([50 50 50])
mu = sum(M ./ vec([3 3 3]),1)

println("Good2")

Mc = M .- mu
Xc, sdev = centerscalematrix(X,M,y.refs)

Sb = Mc' * Mc
Sw = Xc' * Xc

Ub, Db, Ubt = svd(Sb,false)
Uw, Dw, Uwt = svd(Sw,false)

EF = eigfact(Sb,Sw)
A = ((1 ./ sqrt(Dw)) .* Uwt) * (Ub .* (1 ./ sqrt(Db)))
SVD = svdfact(((1 ./ sqrt(Dw)) .* Uwt) * (Ub .* (1 ./ sqrt(Db))))

# inv(Sw) * Sb
# A' * A






