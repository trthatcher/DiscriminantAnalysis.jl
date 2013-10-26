include("DA.jl")

using DA

using DataFrames
using RDatasets



iris = data("datasets", "iris")

clean_colnames!(iris)

pool!(iris, ["Species"])

fm = Formula(:(Species ~ Sepal_Length + Sepal_Width))

y = iris[["Species"]][1]
mf = ModelFrame(fm, iris)

mm = ModelMatrix(mf)

x = rda(fm, iris)

p = [1 2; 3 1; 23.2 1]

#predict(x,p)
