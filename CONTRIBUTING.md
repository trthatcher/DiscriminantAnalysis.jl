# Contributing to DiscriminantAnalysis.jl

## Notation

| Variable | Meaning | Notes |
|:-:|---|---|
| `i` | Iteration index over observations | `for i = 1:n ... end` |
| `j` | Iteration index over predictors | `for j = 1:p ... end` |
| `k` | Iteration index over classes | `for k = 1:m ... end` |
| `nₘ` | Vector of observation counts by class | Ex. `nₘ = [150, 300, 250]`
| `n` | Observation count | `n = sum(nₘ)`
| `m` | Class count | `m = length(nₘ)`
| `p` | Predictor count |
| `X` | Data matrix |
| `Y` | Class label indicator matrix |
| `y` | Class index vector |
| `M` | Class centroid matrix |
| `π` | Class priors vector |
| `Σ` | Covariance matrix |
| `W` | Whitening matrix |
| `Δ` | Discriminant function matrix |