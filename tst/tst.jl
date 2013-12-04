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

mm = ModelMatrix(mf)

x = rda(fm, iris, lambda=0.5)


p = predict(x,x.dp.X)


z = lda(fm, iris)

q = predict(z, z.dp.X)
