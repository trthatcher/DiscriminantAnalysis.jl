Discriminant Analysis
=====================

`DiscriminantAnalysis.jl`_ is a Julia package for multiple linear and quadratic 
regularized discriminant analysis (LDA & QDA respectively). LDA and QDA are
distribution-based classifiers that make the (strong) assumption that the 
underlying data follows a multivariate normal distribution. LDA is distinct from
QDA in the assumption about the class variability; LDA assumes that all classes 
have the same variance whereas QDA allows each class to have its own covariance
matrix. This results in LDA being a linear classifier and QDA being a quadratic
classifier.

.. toctree::
   :maxdepth: 2

.. _DiscriminantAnalysis.jl: https://github.com/trthatcher/DiscriminantAnalysis.jl

Classification
--------------

Linear and Quadratic Discriminant Analysis in the context of classification 
arise as simple probabilistic classifiers. Discriminant Analysis works under the
assumption that each class follows a Gaussian distribution. That is, for each
class :math:`k`, the probability distribution can be modelled by:

.. math::
    
    f_k(x) = \frac{\exp\left((\mathbf{x}-\mathbf{\mu_k})^{\intercal}\Sigma_k^{-1}(\mathbf{x}-\mathbf{\mu_k})\right)}{(2\pi)^{p/2}\left|\Sigma_k\right|^{1/2}}

Let :math:`\pi_k` represent the prior class membership probabilities. 
Application of Baye's Theorem results in:

.. math::

    P(K = k | X = \mathbf{x}) = \frac{f_k(\mathbf{x})\pi_k}{\sum_i f_i(\mathbf{x})\pi_i}

Following rule can be used for classification:

.. math::

    \operatorname{arg\,max}_k\frac{f_k(\mathbf{x})\pi_k}{\sum_i f_i(\mathbf{x})\pi_i}
    = \operatorname{arg\,max}_k f_k(\mathbf{x})\pi_k

Applying the natural logarithm and dropping several constants, the discriminant
functions :math:`\delta_k` may be defined as:

.. math::

    \delta_k(x) =  
    -\frac{1}{2}(\mathbf{x}-\mathbf{\mu_k})^{\intercal}\Sigma_k^{-1}(\mathbf{x}-\mathbf{\mu_k})
    -\frac{1}{2}\log\left(\left|\Sigma_k\right|\right) 
    + \log(\pi_k)

Linear Discriminant Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear Discriminant Analysis works under the additional assumption that
:math:`\Sigma_k = \Sigma` for each class :math:`k`. In other words, the classes
share a common within-class covariance matrix. Since
:math:`\mathbf{x}^\intercal \Sigma \mathbf{x}` is now a constant, this 
simplifies the discriminant function to a linear classifier:

.. math::

    \delta_k(x) =  
    -\mathbf{\mu_k}^{\intercal}\Sigma^{-1}\mathbf{x} +
    \frac{1}{2}\mathbf{\mu_k}\Sigma^{-1}\mathbf{\mu_k}
    + \log(\pi_k)


Canonical Discriminant Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quadratic Discriminant Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
