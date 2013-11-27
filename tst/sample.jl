include("../src/DA.jl")

using DA

using DataFrames
using DataArrays
using RDatasets



iris = data("datasets", "iris")

clean_colnames!(iris)

pool!(iris, ["Species"])

y = iris[["Species"]][1]

train = vec(rand(150,1) .< 0.8)
test = train .== false

fm = Formula(:(Species ~ Sepal_Length + Sepal_Width + Petal_Length + Petal_Width))


# LDA

lda_mod = lda(fm, iris[train,:])
lda_pred = predict(lda_mod,iris[test,:])
100*sum(lda_pred .== y[test])/length(y[test])
scaling(lda_mod)

lda_mod = lda(fm, iris[train,:], gamma=0.2)
lda_pred = predict(lda_mod,iris[test,:])
100*sum(lda_pred .== y[test])/length(y[test])
scaling(lda_mod)

lda_mod = lda(fm, iris[train,:], rrlda=false)
scaling(lda_mod)

lda_mod = lda(fm, iris[train,:], tol=0.1)

# QDA
qda_mod = qda(fm, iris[train,:], gamma=0.1, tol=0.0001)
qda_pred = predict(qda_mod,iris[test,:])
100*sum(qda_pred .== y[test])/length(y[test])

# RDA
rda_mod = rda(fm, iris[train,:], lambda=0.5)
rda_pred = rredict(qda_mod,iris[test,:])
100*sum(rda_pred .== y[test])/length(y[test])

