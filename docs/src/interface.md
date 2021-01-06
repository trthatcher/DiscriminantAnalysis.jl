# Package Interface

<!--
.. _format notes:

.. note::

    Data matrices may be stored in either row-major or column-major ordering of
    observations. Row-major ordering means each row corresponds to an
    observation and column-major ordering means each column corresponds to an
    observation:

    .. math:: \mathbf{X}_{row} = 
                  \begin{bmatrix} 
                      \leftarrow \mathbf{x}_1 \rightarrow \\ 
                      \leftarrow \mathbf{x}_2 \rightarrow \\ 
                      \vdots \\ 
                      \leftarrow \mathbf{x}_n \rightarrow 
                   \end{bmatrix}
              \qquad
              \mathbf{X}_{col} = 
                  \begin{bmatrix}
                      \uparrow & \uparrow & & \uparrow  \\
                      \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x_n} \\
                      \downarrow & \downarrow & & \downarrow
                  \end{bmatrix}

    In DiscriminantAnalysis.jl, the input data matrix ``X`` is assumed to be 
    stored in the same format as a `design matrix`_ in statistics (row-major) by
    default. This ordering can be switched between row-major and column-major by
    setting the ``order`` argument to ``Val{:row}`` and ``Val{:col}``,
    respectively.

.. _design matrix: https://en.wikipedia.org/wiki/Design_matrix

.. function:: lda(X, y [; order, M, priors, gamma])

    Fit a regularized linear discriminant model based on data ``X`` and class 
    identifier ``y``. ``X`` must be a matrix of floats and ``y`` must be a 
    vector of positive integers that index the classes. ``M`` is an optional 
    matrix of class means. If ``M`` is not supplied, it defaults to point 
    estimates of the class means. The ``priors`` argument represents the prior 
    probability of class membership. If ``priors`` is not supplied, it defaults
    to equal class weights.

    .. note::

        See the `format notes`_ for the data matrix ``X``.
    
    Gamma is a regularization parameter that shrinks the covariance matrix 
    towards the average eigenvalue:

    .. math::

        \mathbf{\Sigma}(\gamma) = (1-\gamma)\mathbf{\Sigma} + \gamma
          \left(\frac{\operatorname{trace}(\mathbf{\Sigma})}{p}\right) \mathbf{I}

    This type of regularization can be used counteract bias in the eigenvalue
    estimates generated from the sample covariance matrix.

    The components of the LDA model may be extracted from the ``ModelLDA`` 
    object returned by the ``lda`` function:

    ========== =====================================================
    Field      Description
    ========== =====================================================
    ``is_cda`` Boolean value; the model is a CDA model if ``true``
    ``W``      The whitening matrix used to decorrelate observations
    ``order``  The ordering of observations in the data matrix
    ``M``      A matrix of class means; one per row
    ``priors`` A vector of class prior probabilities
    ``gamma``  The regularization parameter as defined above.
    ========== =====================================================


.. function:: cda(X, y [; order, M, priors, gamma])

    Fit a regularized canonical discriminant model based on data ``X`` and class 
    identifier ``y``. The CDA model is identical to an LDA model, except that
    dimensionality reduction is included in the whitening transformation matrix.
    See the ``lda`` documentation for information on the arguments.

.. function:: qda(X, y [; order, M, priors, gamma, lambda])

    Fit a regularized quadratic discriminant model based on data ``X`` and class 
    identifier ``y``. ``X`` must be a matrix of floats and ``y`` must be a 
    vector of positive integers that index the classes. ``M`` is an optional 
    matrix of class means. If ``M`` is not supplied, it defaults to point 
    estimates of the class means. The ``priors`` argument represents the prior 
    probability of class membership. If ``priors`` is not supplied, it defaults
    to equal class weights.
    
    .. note::

        See the `format notes`_ for the data matrix ``X``.

    Lambda is a regularization parameter that shrinks the class covariance 
    matrices towards the overall covariance matrix:

    .. math::

        \mathbf{\Sigma}_{k}(\lambda) = (1-\lambda)\mathbf{\Sigma}_k 
         + \lambda \mathbf{\Sigma}

    As in LDA, gamma is a regularization parameter that shrinks the covariance
    matrix towards the average eigenvalue:

    .. math::

        \mathbf{\Sigma}_{k}(\gamma,\lambda) 
        = (1-\gamma)\mathbf{\Sigma}_{k}(\lambda) + \gamma
          \left(\frac{\operatorname{trace}(\mathbf{\Sigma}_{k}(\lambda))}{p}\right) \mathbf{I}
     
    The components of the QDA model may be extracted from the ``ModelQDA`` 
    object returned by the ``qda`` function:

    ========== =====================================================
    Field      Description
    ========== =====================================================
    ``W_k``    The vector of whitening matrices (one per class)
    ``order``  The ordering of observations in the data matrix
    ``M``      A matrix of class means; one per row
    ``priors`` A vector of class prior probabilities
    ``gamma``  The regularization parameter as defined above.
    ``lambda`` The regularization parameter as defined above.
    ========== =====================================================

.. function:: discriminants(model, Z)

    Returns a matrix of discriminant function values based on ``model``. Each
    column of values corresponds to a class discriminant function and each row
    corresponds to the discriminant function values for an observation in ``Z``.
    For example, ``Z[i,j]`` corresponds to the discriminant function value of
    class ``j`` for observation ``i``.

.. function:: classify(model, Z)

    Returns a vector of class indices based on the classification rule. This
    function takes the output of the ``discriminants`` function and applies
    ``indmax`` to each row to determine the class.

-->