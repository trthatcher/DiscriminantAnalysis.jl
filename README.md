# Discriminant Analysis

[![Build Status](https://travis-ci.org/trthatcher/DiscriminantAnalysis.jl.svg?branch=master)](https://travis-ci.org/trthatcher/DiscriminantAnalysis.jl)
[![Coverage Status](https://coveralls.io/repos/trthatcher/DiscriminantAnalysis.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/trthatcher/DiscriminantAnalysis.jl?branch=master)

**DiscriminantAnalysis.jl** is a Julia package for multiple linear and quadratic 
regularized discriminant analysis (LDA & QDA respectively). LDA and QDA are
distribution-based classifiers that make the (strong) assumption that the 
underlying data follows a multivariate normal distribution. 

Each class in the sample is fit to a normal distribution with its own centroid.
At classification time, QDA and LDA compute the probability that an observation
belongs to one of the class centroids. The model then assigns the class with the
highest probability of ownership as the prediction. LDA is distinct from QDA in
the assumption about the class variability; LDA assumes that all classes have
the same variance whereas QDA allows each class to have its own covariance
matrix. This results in LDA being a linear classifier and QDA being a quadratic
classifier.

## Getting Started - Linear Discriminant Analysis

## Getting Started - Quadratic Discriminant Analysis
