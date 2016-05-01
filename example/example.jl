using DataFrames, DiscriminantAnalysis

iris_df = readtable("iris.csv")
pool!(iris_df, [:Species])  # Ensure species is made a pooled data vector

X = convert(Array{Float64}, iris_df[[:PetalLength, :PetalWidth, :SepalLength, :SepalWidth]])
y = iris_df[:Species].refs  # Class indices


#== Fitting the LDA model ==#

model1 = lda(X, y)
y_pred1 = classify(model1, X)
accuracy1 = sum(y_pred1 .== y)/length(y)

# The following illustrates column-major ordering with gamma-regularization of 0.1 and non-uniform
# class priors:
model2 = lda(X', y, order = Val{:col}, gamma = 0.1, priors = [2/5; 2/5; 1/5])
y_pred2 = classify(model2, X')
accuracy2 = sum(y_pred2 .== y)/length(y)


#== Fitting the QDA model ==#

model3 = qda(X, y, lambda = 0.1, gamma = 0.1, priors = [1/3; 1/3; 1/3])
y_pred3 = classify(model3, X)
accuracy3 = sum(y_pred3 .== y)/length(y)


#== Fitting the CDA model ==#

model4 = cda(X, y, gamma = Nullable{Float64}(), priors = [1/3; 1/3; 1/3])
y_pred4 = classify(model4, X)
accuracy4 = sum(y_pred4 .== y)/length(y)
