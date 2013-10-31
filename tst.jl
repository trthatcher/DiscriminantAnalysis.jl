include("DA.jl")

using DA

using DataFrames
using RDatasets



iris = data("datasets", "iris")

clean_colnames!(iris)

pool!(iris, ["Species"])

fm = Formula(:(Species ~ Sepal_Length + Sepal_Width + Petal_Length + Petal_Width))

y = iris[["Species"]][1]
mf = ModelFrame(fm, iris)

mm = ModelMatrix(mf)

x = rda(fm, iris)


p = predict(x.dp,x.dp.X)
