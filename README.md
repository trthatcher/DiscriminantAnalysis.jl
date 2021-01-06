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

A bare-bones implementation of LDA is currently available. The script below demonstrates how to fit an LDA model to some synthetic data using the low-level interface:

```julia
using DiscriminantAnalysis
using Random

const DA = DiscriminantAnalysis

# Generate two sets of 100 samples of a 5-dimensional random normal 
# variable offset by +1/-1
X = [randn(250,5) .- 1;
     randn(250,5) .+ 1]

# Generate class labels for the two samples
#   NOTE: classes must be indexed by integers from 1 to the number of 
#         classes (2 in this case)
y = repeat(1:2, inner=100)

# Set the solver options
dims = 1                    # use 1 for row-per-observation; 2 for columns
canonical = true            # use true to compute the canonical coords
compute_covariance = false  # use true to compute & store covariance
centroids = nothing         # supply a precomputed set of class centroids
priors = Float64[1/2; 1/2]  # prior class weights
gamma = nothing             # gamma parameter

# Fit a model
model = DA.LinearDiscriminantModel{Float64}()
DA._fit!(model, y, X, dims, canonical, compute_covariance, centroids, priors, gamma)

# Get the posterior probabilities for new data
Z = rand(10,5) .- 0.5
Z_prob = DA.posteriors(model, Z)
```