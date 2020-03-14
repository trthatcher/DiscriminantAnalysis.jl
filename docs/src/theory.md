# Discriminant Analysis

**Linear Discriminant Analysis** (LDA) and **Quadratic Discriminant Analysis** (QDA) arise as simple probabilistic classifiers. Discriminant Analysis works under the assumption that each class follows a Gaussian distribution. That is, for each class $k$, the probability distribution can be modelled by the function $f$:

$$
f_k(x) = \frac{\exp\left(\frac{-1}{2}(\mathbf{x}-\mathbf{\mu_k})^{\intercal}\Sigma_k^{-1}(\mathbf{x}-\mathbf{\mu_k})\right)}{(2\pi)^{p/2}\left|\Sigma_k\right|^{1/2}}
$$

where $\mu_k \in \mathbb{R}^p$ is the centroid for class $k$, $\Sigma_k \in \mathbb{R}^{p \times p}$ is the covariance matrix for class $k$ and $\mathbf{x} \in \mathbb{R}^p$ is a data vector. Let $\pi_k$ represent the prior class membership probabilities. Application of Baye's Theorem allows us to assign a probability to each class membership given an observation
$\mathbf{x}$:

$$
P(K = k | X = \mathbf{x}) = \frac{f_k(\mathbf{x})\pi_k}{\sum_i f_i(\mathbf{x})\pi_i}
$$

We can take the $\argmax$ over $k$ over the above function as our classification rule to develop a simple classification rule. Noting that the probabilities are non-zero and the natural logarithm is monotonically increasing, the rule can be simplified to:

$$
\argmax_k \left[\ln\left(f_k(\mathbf{x})\right) + \ln(\pi_k) \right]
= \argmax_k \delta_k(\mathbf{x})
$$

The set of functions $\delta_k$ are known as **discriminant functions**. For QDA, the discriminant functions are quadratic in $\mathbf{x}$:

$$
\delta_k(\mathbf{x}) = -\frac{1}{2}\left[
    (\mathbf{x}-\mathbf{\mu_k})^{\intercal}\Sigma_k^{-1}(\mathbf{x}-\mathbf{\mu_k})
    + \ln\left|\Sigma_k\right|
\right] + \ln(\pi_k)
$$

LDA has the additional simplifying assumption that $\Sigma_k = \Sigma \; \forall \; k$. That is, the classes share a common within-class covariance matrix. Since the $\mathbf{x}^\intercal \Sigma \mathbf{x}$ and $\ln|\Sigma|$ terms are constant across classes, they can be eliminated since they have no impact on the classification rule. The discriminant functions then simplify to a linear rule:

$$
\delta_k(x) = 
    - \mathbf{\mu_k}^{\intercal}\Sigma^{-1}\mathbf{x} 
    + \frac{1}{2}\mathbf{\mu_k}\Sigma^{-1}\mathbf{\mu_k}
    + \ln(\pi_k)
$$

## Estimation

To construct a discriminant model, only the class centroids and class covariance matrices (or overall covariance matrix) are required. In practice, these statistics are unknown and must be computed from the data. Given a data sample $\mathcal{D} = \{\mathbf{x}_i \}_{i=1}^n$, a data matrix $\mathbf{X}$ is a matrix of observations that are stored as either as columns or as rows:

$$
\mathbf{X}^{\text{col}} = \begin{bmatrix}
    \uparrow & \uparrow & & \uparrow \\
    \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_n \\
    \downarrow & \downarrow & & \downarrow
\end{bmatrix}
\quad \text{and} \quad
\mathbf{X}^{\text{row}} = \begin{bmatrix}
    \leftarrow & \mathbf{x}_1^\intercal & \rightarrow \\
    \leftarrow & \mathbf{x}_2^\intercal & \rightarrow \\
    & \vdots & \\
    \leftarrow & \mathbf{x}_n^\intercal & \rightarrow
\end{bmatrix}
$$

While the orientation of the data matrix is largely an implementation detail, most statistics textbooks use the row format in their formulas and derivations. However, this has the disadvantage of data vectors being transposed relative to the random vector distribution they were sampled from. 

For a class $k$ with corresponding sample $\mathcal{D}_k = \{\mathbf{x}_{i} \}_{i=1}^{n_k}$, we  can compute an estimate of the class centroid using the standard formula for an average and a corresponding centered sample:

$$
\hat{\mu}_k = \bar{\mathbf{x}}_k = \frac{1}{n}\sum_{i=1}^{n_k} \mathbf{x}_i
\quad \implies \quad
\tilde{\mathcal{D}}_k = \{\mathbf{x}_{i} - \hat{\mu}_k \}_{i=1}^{n_k}
= \{\tilde{\mathbf{x}}_{i}\}_{i=1}^{n_k}
$$

Given a centered data matrix $\tilde{\mathbf{X}}_k$ that consists of observations from sample $\tilde{\mathcal{D}}_k$, the class covariance matrix estimate can be computed as follows:

$$
\hat{\Sigma}
= \frac{1}{n_k-1} \sum_{i=1}^{n_k} \tilde{\mathbf{x}}_i\tilde{\mathbf{x}}_i^\intercal
= \begin{cases}
\frac{1}{n-1} \tilde{\mathbf{X}} \tilde{\mathbf{X}}^\intercal
&\text{if column oriented}\\
\frac{1}{n-1} \tilde{\mathbf{X}}^\intercal \tilde{\mathbf{X}}
&\text{if row oriented}

\end{cases}
$$

## Regularization

Section to be completed.

## Data Whitening

One main issue with the discriminant rules outlined above is that each discriminant function requires computing the inverse of the covariance matrix. Computing a matrix inverse is a computationally intensive procedure. Further, full inversion of a matrix may introduce numerical error as a result of floating point arithmetic. Therefore, data is typically "sphered" or "whitened" to decorrelate predictors and simplify the discriminant functions while maintaining numerical stability. A whitening transform for a random vector $\mathbf{x}$ with covariance matrix $\Sigma$ is a matrix $\mathbf{W}$ such that:

$$
\operatorname{Var}(\mathbf{Wx})
= \mathbf{W} \Sigma \mathbf{W}^\intercal = \mathbf{I}
\quad \implies \quad
\Sigma^{-1} = \mathbf{W}^\intercal \mathbf{W}
$$

In general, the whitening transform is not unique based on the characterization above. Two common methods to compute a whitening transform are through the eigendecomposition or the Cholesky decomposition, respectively:

$$
\Sigma = \mathbf{V} \Lambda \mathbf{V}^\intercal 
\quad\text{or}\quad 
\Sigma = \mathbf{L}\mathbf{L}^\intercal
$$

For the eigendecomposition, $\mathbf{V}$ is an orthogonal matrix consisting of the eigenvectors of $\Sigma$ and $\Lambda$ is a positive diagonal matrix of eigenvalues. For the Cholesky decomposition, $\mathbf{L}$ is a lower triangle matrix. This yields two approaches to the whitening transform:

$$
\mathbf{W}^{\text{SVD}} = \Lambda^{-\frac{1}{2}}\mathbf{V}^\intercal
\quad \text{and} \quad
\mathbf{W}^{\text{Chol}} = \mathbf{L}^{-1}
$$

In general, computing and inverting the Cholesky decomposition has a lower runtime complexity than computing the eigendecomposition so the Cholesky method is the preferred approach.

For QDA, a whitening matrix is defined for each class covariance matrix $\Sigma_k$. For a specific class $k$, we can define the transformed random vector of $\mathbf{x}$ and transformed class mean of $\mu$ as:

$$
\tilde{\mathbf{x}}_k = \mathbf{W}_k \mathbf{x}
\quad\text{and}\quad
\tilde{\mathbf{\mu}}_k = \mathbf{W}_k \mu_k
$$

Since $\Sigma_k^{-1} = \mathbf{W}_k^\intercal \mathbf{W}_k$, this simplifies the quadratic term of the discriminant function:

$$
(\mathbf{x}-\tilde{\mathbf{\mu}}_k)^{\intercal}\Sigma_k^{-1}(\mathbf{x}-\tilde{\mathbf{\mu}}_k)
= (\tilde{\mathbf{x}}_k-\tilde{\mathbf{\mu}}_k)^{\intercal}(\tilde{\mathbf{x}}_k-\tilde{\mathbf{\mu}}_k)
= ||\tilde{\mathbf{x}}_k-\tilde{\mathbf{\mu}}_k||_2^2
$$

Therefore, under the whitening transform, the QDA discriminant functions simplify to:

$$
\delta_k(\mathbf{x}) = -\frac{1}{2} \left[
    ||\mathbf{z}-\tilde{\mathbf{\mu}}_k||_2^2
    + \ln\left|\Sigma_k\right|
\right] + \ln(\pi_k)
$$

Similarly, the LDA discriminant functions simplify to:

$$
\delta_k(\mathbf{x}) =
-\frac{1}{2} ||\mathbf{z}-\tilde{\mathbf{\mu}}_k||_2^2 + \ln(\pi_k)
$$

## Canonical Discriminant Analysis (CDA)

Canonical discriminant analysis expands upon linear discriminant analysis by noting that 
the class centroids lie in a $m-1$ dimensional subspace of the $p$ dimensions of the data 
where $m$ is the number of classes. Defining the overall centroid and the between-class 
covariance matrix:

$$
\mu = \sum_{k=1}^m \pi_k \mu_k
\qquad \text{and} \qquad
\Sigma_b = \sum_{k=1}^{m} \pi_k (\mu_k - \mu)(\mu_k - \mu)^{\intercal}
$$

Note that although $\mathbf{\Sigma}$ is a full rank matrix by design, the rank of $\mathbf{\Sigma}_b$ is $m-1$. 

The goal of canonical discriminant analysis is to find the vector that maximizes the class
separation. This corresponds to maximizing the generalized Rayleigh quotient:

$$
\mathbf{c}^{\intercal} \mathbf{x}
\quad \text{where} \quad
\mathbf{c} = 
\argmax_{\mathbf{u} \in \mathbb{R}^{p}}
\frac{\mathbf{u}^{\intercal}\Sigma_b\mathbf{u}}{\mathbf{u}^{\intercal}\Sigma\mathbf{u}}
$$

The vector $\mathbf{c}$ is known as a **canonical coordinate**. The problem can also be extended to a multiclass case:

$$
\mathbf{C} \mathbf{x}
\quad \text{where} \quad
\mathbf{C} =
\argmax_{\mathbf{U} \in \mathbb{R}^{m-1\times p}}\frac{\left|\mathbf{U} \Sigma_b \mathbf{U}^\intercal\right|}{\left|\mathbf{U} \Sigma \mathbf{U}^\intercal\right|}
$$

where the rows of $\mathbf{C}$ are the $m-1$ canonical coordinates. It has been shown that the solution to the equation above corresponds to the generalized eigenvectors of $\Sigma_b$ and $\Sigma$:

$$
\Sigma_b \mathbf{V} = \Sigma \mathbf{V} \Lambda
\quad \implies \quad
\mathbf{C} = \mathbf{V}^{\intercal}
$$

where $\mathbf{V}$ is a matrix of generalized eigenvectors (columns), $\Lambda$ is a positive diagonal matrix of generalized eigenvalues. Since $\mathbf{\Sigma}$ is a positive definite matrix by design, we can transform the generalized eigendecomposition into a regular eigendecomposition using the following procedure:

$$
\Sigma_b \mathbf{V} = (\mathbf{W}^{\intercal}\mathbf{W})^{-1} \mathbf{V} \Lambda
\quad \implies \quad
\mathbf{W} \Sigma_b \mathbf{V} = \mathbf{W}^{-\intercal} \mathbf{V} \Lambda
$$

Let $\tilde{\mathbf{V}} = \mathbf{W}^{-\intercal}\mathbf{V}$ so that $\mathbf{V} = \mathbf{W}^\intercal \tilde{\mathbf{V}}$, then substitute $\tilde{\mathbf{V}}$ into the above equation:

$$
\mathbf{W} \Sigma_b \left(\mathbf{W}^\intercal \tilde{\mathbf{V}}\right)
= \mathbf{W}^{-\intercal} \left(\mathbf{W}^\intercal \tilde{\mathbf{V}} \right) \Lambda
\quad \implies \quad
\mathbf{W} \Sigma_b \mathbf{W}^\intercal \tilde{\mathbf{V}}
= \tilde{\mathbf{V}} \Lambda
$$

Now, $\tilde{\mathbf{V}}$ can be solved for by decomposing $\mathbf{W} \Sigma_b \mathbf{W}^\intercal$ using an ordinary eigensolver. We can compute $\mathbf{C}$ by inverting the transformations above:

$$
\tilde{\mathbf{V}} = \mathbf{W}^{-\intercal}\mathbf{C}^\intercal
\quad \implies \quad
\mathbf{C} = \tilde{\mathbf{V}}^\intercal \mathbf{W}
$$

## Implementation Best Practice

For row data, the whitening matrix must be right applied

$$
\mathbf{X}_{\text{r}} = \mathbf{Q}\mathbf{D}\mathbf{V}^\intercal
\quad \implies \quad
\mathbf{W}^{\text{SVD}} = \mathbf{V} \mathbf{D}^{-1}
$$


Using the QR decomposition, the Cholesky whitening matrix can be computed
$$
\mathbf{X}_{\text{r}} = \mathbf{Q}\mathbf{R}
\quad \implies \quad
\mathbf{W}^{\text{Chol}} = \mathbf{R}^{-1}
$$

Similarly for column data, the whitening matrices must be left-applied

$$
\mathbf{X}_{\text{c}} = \mathbf{Q}\mathbf{D}\mathbf{V}^\intercal
\quad \implies \quad
\mathbf{W}^{\text{SVD}} = \mathbf{D}^{-1} \mathbf{Q}^\intercal 
$$

$$
\mathbf{X}_{\text{c}} = \mathbf{L}\mathbf{Q}
\quad \implies \quad
\mathbf{W}^{\text{Chol}} = \mathbf{L}^{-1}
$$

Data matrices

QR
SVD

## References

  * Friedman J. 1989. *Regularized discriminant analysis.* Journal of the American statistical association 84.405; p. 165-175.
  * Hastie T, Tibshirani R, Friedman J, Franklin J. 2005. *The elements of statistical learning: data mining, inference and prediction*. The Mathematical Intelligencer, 27(2); p. 83-85.
