# Contributing to DiscriminantAnalysis.jl

## Notation

| Variable | Meaning |
|:-:|---|
| `i` | Iteration index over observations |
| `j` | Iteration index over predictors |
| `k` | Iteration index over classes |
| `m` | Class count |
| `nₘ` | Vector of observation counts by class
| `n` | Observation count. `n = sum(nₘ)` |
| `p` | Predictor count |
| `X` | Data matrix |
| `Y` | Class label indicator matrix |
| `y` | Class index vector |
| `M` | Class centroid matrix |
| `π` | Class priors vector |
| `Σ` | Covariance matrix
| `W` | Whitening matrix