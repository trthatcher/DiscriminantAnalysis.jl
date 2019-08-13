# Introduction

**DiscriminantAnalysis.jl** is a Julia package for multiple linear and quadratic 
regularized discriminant analysis (LDA & QDA respectively). LDA and QDA are
distribution-based classifiers with the underlying assumption that data follows
a multivariate normal distribution. LDA differs from QDA in the assumption about 
the class variability; LDA assumes that all classes share the same within-class 
covariance matrix whereas QDA relaxes that constraint and allows for distinct 
within-class covariance matrices. This results in LDA being a linear classifier
and QDA being a quadratic classifier.

## Discriminant Analysis

See the [theory section](theory.md) of the documentation for an overview of the theory surrounding classification via discriminant analysis.

## Installation

To add the package from the Julia REPL, enter the package manager with `]`:

```bash
(v1.1) pkg> add DiscriminantAnalysis
```

## Package API

See the [interface section](interface.md) of the documentation for an overview of the package interface.


## Source Code

The source code is available on Github:

  > https://github.com/trthatcher/DiscriminantAnalysis.jl
