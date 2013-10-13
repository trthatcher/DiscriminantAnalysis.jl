using DA
using DataFrames
using RDatasets

iris = data("datasets", "iris")

clean_colnames!(iris)

pool!(iris, ["Species"])

fm = Formula(:(Species ~ Sepal_Length + Sepal_Width))

x = rda(fm, iris)
