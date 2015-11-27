using DataFrames

iris_df = readtable("iris.csv")
pool!(iris_df, [:species])  # Ensure species is made a pooled data vector

X = convert(Array{Float64}, iris_df[[:petal_length, :petal_width, :sepal_length, :sepal_width]])
y = iris_df[:species].refs  # Class indices

using DiscriminantAnalysis

model = qda(X, y, lambda = 0.3, gamma = 0.0, priors = [1/3; 1/3; 1/3])

y_pred = DiscriminantAnalysis.predict(model, X)  # Note: DataFrames exports a predict() function

accuracy = sum(y_pred .== y)/length(y)
