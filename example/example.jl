using DataFrames

iris_df = readtable("iris.csv")
pool!(iris_df, [:Species])  # Ensure species is made a pooled data vector

X = convert(Array{Float64}, iris_df[[:PetalLength, :PetalWidth, :SepalLength, :SepalWidth]])
y = iris_df[:Species].refs  # Class indices

using DiscriminantAnalysis

model1 = qda(X, y, lambda = 0.3, gamma = 0.0, priors = [1/3; 1/3; 1/3])

y_pred1 = classify(model1, X)

accuracy1 = sum(y_pred1 .== y)/length(y)

model2 = lda(X, y, gamma = 0.0, priors = [1/3; 1/3; 1/3])

y_pred2 = classify(model2, X)

accuracy2 = sum(y_pred2 .== y)/length(y)

model3 = cda(X, y, gamma = 0.0, priors = [1/3; 1/3; 1/3])

y_pred3 = classify(model3, X)

accuracy3 = sum(y_pred3 .== y)/length(y)
