# Discriminant Analysis
=======================

This package is for linear and quadratic regularized discriminant analysis.

The dataframes packages is required as all of the methods accept dataframes and formulae: 

```julia
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
```

## Linear Discriminant Analysis
A linear discriminant classifier can be built using the `lda` function and a dataframe. Here I am using the iris data set that I have divided into a training set (to build the classifier) and a testing set to validate against:

```julia
julia> lda_mod = lda(fm, iris[train,:])
Formula: Species ~ :(+(Sepal_Length,Sepal_Width,Petal_Length,Petal_Width))

Response:

3x3 DataFrame:
               Group    Prior Count
[1,]        "setosa" 0.333333    37
[2,]    "versicolor" 0.333333    37
[3,]     "virginica" 0.333333    38


Gamma: 0
Rank-reduced: true

Class means:
3x5 DataFrame:
               Group Sepal_Length Sepal_Width Petal_Length Petal_Width
[1,]        "setosa"      5.01622     3.48649      1.46757    0.243243
[2,]    "versicolor"      5.87297     2.77568      4.23514     1.32973
[3,]     "virginica"      6.68158     2.98158      5.59474     2.03421
```

By default, rank-reduced linear discriminant analysis is performed. This (probably) will perform a dimensionality reduction if there is more than two groups (similar to principle components analysis). 

The scaling matrix is used to "sphere" or "whiten" the input data so that its sample covariance matrix is the identity matrix (this decreases the complexity of the classification computation). In other words, the whitened data has a sample covariance that corresponds to the unit n-sphere.

Note: rank reduction was successful so the scaling matrix is of rank two rather than three. 

```julia
julia> scaling(lda_mod)
4x2 Array{Float64,2}:
  0.660804   0.849652
  1.59634    1.81788 
 -1.87905   -0.978034
 -2.85134    2.14334 
```

Prediction is as simple as plugging a dataframe and the model into the predict function. The model will extract the appropriate columns from the dataframe assuming they are named correctly:

```julia
julia> lda_pred = predict(lda_mod,iris[test,:])
38x1 PooledDataArray{UTF8String,Uint32,2}:
 "setosa"    
 ⋮          
 "virginica"

julia> 100*sum(lda_pred .== y[test])/length(y[test])
100.0
```

Regularized linear discriminant analysis has an additional parameter `gamma`. This regularization is analogous to ridge regression and can be used to 'nudge' a singular matrix into a non singular matrix (or help penalize the biased estimates of the eigenvalues - see paper below). This is important when the sample size is small and the sample covariance matrix may not be invertible. 

The `gamma` values supplied should be between 0 and 1 inclusive. The value represents the percentage of shrinkage along the diagonals of the sample covariance matrix towards its average eigenvalue.

```julia
julia> lda_mod = lda(fm, iris[train,:], gamma=0.2)

julia> scaling(lda_mod)
4x2 Array{Float64,2}:
 -0.122872   0.39509 
  0.554429   1.50014 
 -0.938699  -0.282481
 -1.70349    0.797025
```

Rank-reduction can be disabled setting the parameter `rrlda` to `false`. Default is `true`. When it is disabled, we can see the scaling matrix is square:

```julia
julia> lda_mod = lda(fm, iris[train,:], rrlda=false)

julia> scaling(lda_mod)
4x4 Array{Float64,2}:
 -0.708728   0.919018  -0.970648   2.99623
 -0.85916   -2.03842   -2.20533   -2.15067
 -0.76332    1.29499    0.677356  -3.26619
 -1.38388   -2.33625    4.27793    2.95553
```

Lastly, a tolerance parameter can be set and is used in determining the rank of all covariance matrices. It is relative to the largest eigenvalue of the sample covariance matrix and should be between 0 and 1. 

```julia
julia> lda_mod = lda(fm, iris[train,:], tol=0.1)
ERROR: Rank deficiency detected with tolerance=0.1.
 in error at error.jl:21
```

## Quadratic Discriminant Analysis
----------------------------------

```julia
julia> qda_mod = qda(fm, iris[train,:], gamma=0.1, tol=0.0001)
Formula: Species ~ :(+(Sepal_Length,Sepal_Width,Petal_Length,Petal_Width))

Response:

3x3 DataFrame:
               Group    Prior Count
[1,]        "setosa" 0.333333    37
[2,]    "versicolor" 0.333333    37
[3,]     "virginica" 0.333333    38


Gamma: 0.1

Class means:
3x5 DataFrame:
               Group Sepal_Length Sepal_Width Petal_Length Petal_Width
[1,]        "setosa"      5.01622     3.48649      1.46757    0.243243
[2,]    "versicolor"      5.87297     2.77568      4.23514     1.32973
[3,]     "virginica"      6.68158     2.98158      5.59474     2.03421



julia> qda_pred = predict(qda_mod,iris[test,:])
38x1 PooledDataArray{UTF8String,Uint32,2}:  
 "setosa"   
 ⋮          
 "virginica"

julia> 100*sum(qda_pred .== y[test])/length(y[test])
100.0

julia> scaling(qda_mod)
4x4x3 Array{Float64,3}:
[:, :, 1] =
 -0.718864   0.225921  -2.21662    1.08868 
 -1.87503   -0.911666   1.57226   -0.984448
 -0.10516    1.58782   -0.841958  -2.53646 
 -0.554306   4.69676    2.47484    2.86637 

[:, :, 2] =
 -0.540377   0.689292  -1.69643   -0.712444
 -0.648636  -2.67637   -0.628192   0.760625
 -0.741854   0.837311   0.729449   1.83986 
 -1.25651   -0.245052   2.69873   -3.94103 

[:, :, 3] =
 0.633468  -0.535131   0.304746   1.55392 
 0.438729   0.726428   2.40918   -0.750259
 0.716968  -0.576765  -0.461987  -1.65265 
 1.13231    2.37764   -1.89922    0.656749
```

## Regularized Discriminant Analysis
------------------------------------

```julia
julia> rda_mod = rda(fm, iris[train,:], lambda=0.5)
Formula: Species ~ :(+(Sepal_Length,Sepal_Width,Petal_Length,Petal_Width))

Response:

3x3 DataFrame:
               Group    Prior Count
[1,]        "setosa" 0.333333    37
[2,]    "versicolor" 0.333333    37
[3,]     "virginica" 0.333333    38


Lambda: 0.5
Gamma: 0

Class means:
3x5 DataFrame:
               Group Sepal_Length Sepal_Width Petal_Length Petal_Width
[1,]        "setosa"      5.01622     3.48649      1.46757    0.243243
[2,]    "versicolor"      5.87297     2.77568      4.23514     1.32973
[3,]     "virginica"      6.68158     2.98158      5.59474     2.03421



julia> rda_pred = predict(rda_mod,iris[test,:])
38x1 PooledDataArray{UTF8String,Uint32,2}:
 "setosa"     
 ⋮          
 "virginica"

julia> 100*sum(rda_pred .== y[test])/length(y[test])
100.0

```

## References
Jerome H. Friedman, Regularized Discriminant Analysis, Journal of the American Statistical Association. Vol. 84, No. 405 (Mar., 1989), pp. 165-175

Trevor Hastie, Robert Tibshirani, and Jerome Friedman, The Elements of Statistical Learning. Springer Series in Statistics Springer New York Inc., New York, NY, USA, (2001)

