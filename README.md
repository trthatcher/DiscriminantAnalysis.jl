# Discriminant Analysis

[![Build Status](https://travis-ci.org/trthatcher/DiscriminantAnalysis.jl.svg?branch=master)](https://travis-ci.org/trthatcher/DiscriminantAnalysis.jl)
[![Coverage Status](https://coveralls.io/repos/trthatcher/DiscriminantAnalysis.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/trthatcher/DiscriminantAnalysis.jl?branch=master)

**DiscriminantAnalysis.jl** is a Julia package for multiple linear and quadratic 
regularized discriminant analysis (LDA & QDA respectively). LDA and QDA are
distribution-based classifiers with the underlying assumption that data follows
a multivariate normal distribution. LDA differs from QDA in the assumption about 
the class variability; LDA assumes that all classes share the same within-class 
covariance matrix whereas QDA relaxes that constraint and allows for distinct 
within-class covariance matrices. This results in LDA being a linear classifier
and QDA being a quadratic classifier.

The package is currently a work in progress work in progress - see [issue #12](https://github.com/trthatcher/DiscriminantAnalysis.jl/issues/12) for the package status.

## Getting Started

A bare-bones implementation of LDA is currently available but is not exported. Calls to the solver must be prefixed with `DiscriminantAnalysis` after running `using DiscriminantAnalysis`. Below is a brief overview of the API:

  * `lda(X, y; kwargs...)`: construct a Linear Discriminant Analysis model.
    * `X`: the matrix of predictors (design matrix). Data may be per-column or per-row; this is specified by the `dims` keyword argument.
    * `y`: the vector of class indices. For c classes, the values must range from 1 to c.
    * `dims=1`: the dimension along which observations are stored. Use 1 for row-per-observation and 2 for column-per-observation.
    * `canonical=false`: compute the canonical coordinates if true. For c classes, the data is mapped to a c-1 dimensional space for prediction.
    * `compute_covariance=false`: compute the full class covariance matrix if true. Data is whitened prior to compute discriminant values, so generally the covariance is not computed unless specified.
    * `centroids=nothing`: matrix of pre-computed class centroids. This can be used if the class centroids are known a priori. Otherwise, the centroids are estimated from the data. The centroid matrix must have the same orientation as specified by the `dims` argument.
    * `priors=nothing`: vector of pre-computed class prior probabilities. This can be used if the class prior probabilities are known a priori. Otherwise, the priors are estimated from the class frequencies.
    * `gamma=nothing`: real value between 0 and 1. Gamma is a regularization parameter that is used to shrink the covariance matrix towards an identity matrix scaled by the average eigenvalue of the covariance matrix. A value of `0.2` retains 80% of the original covariance matrix.
  * `posteriors(LDA, Z)`: compute the class posterior probabilities on a new matrix of predictors `Z`. This matrix must have the same `dims` orientation as the original design matrix `X`.
  * `classify(LDA, Z)`: compute the class label predictions on a new matrix of predictors `Z`. This matrix must have the same `dims` orientation as the original design matrix `X`.


The script below demonstrates how to fit an LDA model to some synthetic data using the interface described above:

```julia
using DiscriminantAnalysis
using Random

const DA = DiscriminantAnalysis

# Generate two sets of 100 samples of a 5-dimensional random normal 
# variable offset by +1/-1
X = [randn(250,5) .- 1;
     randn(250,5) .+ 1];

# Generate class labels for the two samples
#   NOTE: classes must be indexed by integers from 1 to the number of 
#         classes (2 in this case)
y = repeat(1:2, inner=250);

# Construct the LDA model
model = DA.lda(X, y; dims=1, canonical=true, priors=[0.5; 0.5])

# Generate some new data
Z = rand(10,5) .- 0.5

# Get the posterior probabilities for new data
Z_prob = DA.posteriors(model, Z)

# Get the class predictions
Z_class = DA.classify(model, Z)
```