# Quadratic Discriminant Analysis

mutable struct QuadraticDiscriminantModel{T} <: DiscriminantModel{T}
    "Standard discriminant model parameters"
    Θ::DiscriminantParameters{T}
    "Shrinkage parameter towards the common covariance matrix"
    λ::Union{Nothing,T}
    "Covariance matrix for each class"
    Σₘ::Union{Nothing,Vector{AbstractMatrix{T}}}
    "Whitening transformation by class"
    Wₘ::Vector{AbstractMatrix{T}}
    "Discriminant function intercept for each class"
    δ₀::Vector{T}
    function QuadraticDiscriminantModel{T}() where T
        new{T}(DiscriminantParameters{T}(), nothing, Array{T}(undef,0,0,0), 
               Vector{T}(undef,0))
    end
end

function fit!(QDA::QuadraticDiscriminantModel{T},
              y::Vector{<:Integer},
              X::Matrix{T};
              dims::Integer=1,
              canonical::Bool=false,
              store_cov::Bool=false,
              store_class_covs::Bool=false,
              centroids::Union{Nothing,AbstractMatrix}=nothing, 
              priors::Union{Nothing,AbstractVector}=nothing,
              gamma::Union{Nothing,Real}=nothing,
              lambda::Union{Nothing,Real}=nothing) where T
    Θ = QDA.Θ

    if lambda !== nothing
        0 ≤ lambda ≤ 1 || throw(DomainError(lambda, "λ must be in the interval [0,1]"))
    end
    QDA.λ = lambda
    
    set_dimensions!(Θ, y, X, dims)
    set_gamma!(Θ, gamma)
    set_statistics!(Θ, y, X, centroids)
    set_priors!(Θ, priors)

    # Compute overall degrees of freedom
    df = size(X, dims) - Θ.m

    if dims == 1
        # Overall centroid is prior-weighted average of class centroids
        Θ.μ = vec(transpose(Θ.π)*Θ.M)
        # Center the data matrix with respect to classes to compute whitening matrix
        X .-= view(Θ.M, y, :)
        # Compute scatter matrix if requested
        S = (store_cov || QDA.λ !== nothing) ? transpose(X)*X : nothing
    else
        Θ.μ = Θ.M*Θ.π
        X .-= view(Θ.M, :, y)
        S = (store_cov || QDA.λ !== nothing) ? X*transpose(X) : nothing
    end

    # Set up some references
    Σₘ = QDA.Σₘ
    Wₘ = QDA.Wₘ
    δ₀ = QDA.δ₀

    # Allocate space for each covariance matrix if required
    QDA.Σₘ = store_class_covs ? [zeros(T, 0, 0) for k=1:QDA.m] : nothing

    # QDA - find whitening matrix for each class
    for (k, nₖ) in enumerate(Θ.nₘ)
        Xₖ = dims == 1 ? X[y .== k, :] : X[:, y .== k]
        dfₖ = nₖ - 1

        if store_class_covs || QDA.λ !== nothing
            Sₖ = dims == 1 ? transpose(Xₖ)*Xₖ : Xₖ*transpose(Xₖ)

            if QDA.λ === nothing
                # Compute covariance matrix using within-class degrees of freedom
                Σₖ = lmul!(one(T)/dfₖ, Sₖ)
                Σₘ[k] = deepcopy(Σₖ)

                # Compute whitening
                Wₘ[k], δ₀[k] = whiten_cov_chol!(Σₖ, Θ.γ, dims=dims)
            else
                if store_class_covs
                    QDA.Σₘ[k] = lmul!(one(T)/dfₖ, deepcopy(Sₖ))
                end

                # Regularize covariance matrix
                scale = one(T)/regularize(dfₖ, df, QDA.λ)  # denominator is separate
                Σₖ = lmul!(scale, broadcast!(regularize, Sₖ, Sₖ, S, QDA.λ))

                # Compute whitening 
                Wₘ[k], δ₀[k] = whiten_cov_chol!(Σₖ, Θ.γ, dims=dims)
            end
        else
            if Θ.γ === nothing
                Wₘ[k], δ₀[k] = whiten_data_chol!(Xₖ, dims=dims, df=dfₖ)
            else
                Wₘ[k], δ₀[k] = whiten_data_svd!(Xₖ, Θ.γ, dims=dims, df=dfₖ)
            end
        end
    end

    if store_cov
        Θ.Σ = lmul!(one(T)/df, S)
    end

    Θ.fit = true

    return LDA
end

# discriminants

#==========================================================================
  Regularized Quadratic Discriminant Analysis Solver
==========================================================================#

#immutable ModelQDA{T<:BlasReal}
#    order::Union{Type{Val{:row}},Type{Val{:col}}}
#    W_k::Vector{AbstractMatrix{T}}  # Vector of class whitening matrices
#    M::Matrix{T}                    # Matrix of class means (one per row)
#    priors::Vector{T}               # Vector of class priors
#    gamma::Nullable{T}
#    lambda::Nullable{T}
#end
#
#function show(io::IO, model::ModelQDA)
#    println(io, "Quadratic Discriminant Model")
#    println(io, "\nRegularization Parameters:")
#    print(io, "    γ = ")
#    isnull(model.gamma) ? print(io, "N/A") : showcompact(get(model.gamma))
#    print(io, "\n    λ = ")
#    isnull(model.lambda) ? print(io, "N/A") : showcompact(get(model.lambda))
#    println(io, "\n\nClass Priors:")
#    for i in eachindex(model.priors)
#        print(io, "    Class ", i, " Probability: ")
#        showcompact(io, model.priors[i]*100)
#        println(io, "%")
#    end
#    println(io, "\nClass Means:")
#    for j in eachindex(model.priors)
#        print(io, "    Class ", j, " Mean: [")
#        print(io, join([sprint(showcompact, v) for v in subvector(model.order, model.M, j)], ", "))
#        println(io, "]")
#    end
#    print(io, "\n")
#end
#
#function covmatrix{T<:BlasReal}(::Type{Val{:row}}, H::Matrix{T})
#    n = size(H,1)
#    Σ = H'H
#    broadcast!(/, Σ, Σ, n-1)
#end
#
#function covmatrix{T<:BlasReal}(::Type{Val{:col}}, H::Matrix{T})
#    n = size(H,2)
#    Σ = A_mul_Bt(H, H)
#    broadcast!(/, Σ, Σ, n-1)
#end
#
#for (scheme, dim_obs) in ((:(:row), 1), (:(:col), 2))
#    isrowmajor = dim_obs == 1
#    dim_param = isrowmajor ? 2 : 1
#
#    H_i      = isrowmajor ? :(H[y .== i, :]) : :(H[:, y .== i])
#    _ij, _ji = isrowmajor ? (:i, :j) : (:j, :i)  # Swapped variables for row and column ordering
#    _nk, _kn = isrowmajor ? (:n, :k) : (:k, :n)
#    _np, _pn = isrowmajor ? (:n, :p) : (:p, :n)
#    W_j, H   = isrowmajor ? (:(W_k[j]), :H) : (:H, :(W_k[j]))
#
#    @eval begin
#        # λ-regularized QDA - require full covariance matrices
#        function classwhiteners!{T<:BlasReal,U}(
#                 ::Type{Val{$scheme}},
#                H::Matrix{T},
#                y::RefVector{U}, 
#                γ::Nullable{T},
#                λ::T
#            )
#            k = convert(Int64, y.k)
#            Σ_k = AbstractMatrix{T}[covmatrix(Val{$scheme}, $H_i) for i = 1:k]
#            Σ   = covmatrix(Val{$scheme}, H)
#            for i in eachindex(Σ_k)
#                regularize!(Σ_k[i], λ, Σ)
#                Σ_k[i] = whitencov_chol!(Val{$scheme}, Σ_k[i], γ)
#            end
#            Σ_k
#        end
#
#        # No λ-regularization - only need data matrices
#        function classwhiteners!{T<:BlasReal,U}(
#                 ::Type{Val{$scheme}},
#                H::Matrix{T},
#                y::RefVector{U},
#                γ::Nullable{T}
#            )
#            k = convert(Int64, y.k)
#            if isnull(γ)
#                AbstractMatrix{T}[whitendata_qr!(Val{$scheme}, $H_i) for i = 1:k]
#            else
#                AbstractMatrix{T}[whitendata_svd!(Val{$scheme}, $H_i, get(γ)) for i = 1:k]
#            end
#        end
#
#        function qda!{T<:BlasReal,U}(
#                 ::Type{Val{$scheme}},
#                X::Matrix{T}, 
#                M::Matrix{T}, 
#                y::RefVector{U}, 
#                γ::Nullable{T},
#                λ::Nullable{T}
#            )
#            H = centerclasses!(Val{$scheme}, X, M, y)
#            if isnull(λ)
#                classwhiteners!(Val{$scheme}, H, y, γ)
#            else
#                classwhiteners!(Val{$scheme}, H, y, γ, get(λ))
#            end
#        end
#
#        function discriminants_qda{T<:BlasReal}(
#                 ::Type{Val{$scheme}},
#                W_k::Vector{AbstractMatrix{T}},
#                M::Matrix{T},
#                priors::Vector{T},
#                Z::Matrix{T}
#            )
#            n = size(Z, $dim_obs)
#            p = size(Z, $dim_param)
#            k = length(priors)
#            D   = Array(T, $_nk, $_kn)  # discriminant function values
#            H   = Array(T, $_np, $_pn)  # temporary array to prevent re-allocation k times
#            hᵀh = Array(T, n)           # diag(H'H)
#            for j = 1:k
#                broadcast!(-, H, Z, subvector(Val{$scheme}, M, j))
#                dotvectors!(Val{$scheme}, $H * $W_j, hᵀh)
#                for i = 1:n
#                    D[$_ij, $_ji] = -hᵀh[i]/2 + log(priors[j])
#                end
#            end
#            D
#        end
#    end
#end
#
#doc"`qda(X, y; order, M, lambda, gamma, priors)` Fits a regularized quadratic discriminant model to
#the data in `X` based on class identifier `y`."
#function qda{T<:BlasReal,U<:Integer}(
#        X::Matrix{T},
#        y::AbstractVector{U};
#        order::Union{Type{Val{:row}},Type{Val{:col}}} = Val{:row},
#        M::Matrix{T} = classmeans(order, X, RefVector(y)),
#        gamma::Union{T,Nullable{T}}  = Nullable{T}(),
#        lambda::Union{T,Nullable{T}} = Nullable{T}(),
#        priors::Vector{T} = ones(T,maximum(y))/maximum(y)
#    )
#    all(priors .> 0) || error("Argument priors must have positive values only")
#    isapprox(sum(priors), one(T)) || error("Argument priors must sum to 1")
#    yref = isa(y,RefVector) ? y : RefVector(y)
#    length(priors) == yref.k || throw(DimensionMismatch("Prior length must match class count"))
#    γ = isa(gamma,  Nullable) ? gamma  : Nullable(gamma)
#    λ = isa(lambda, Nullable) ? lambda : Nullable(lambda)
#    W_k = qda!(order, copy(X), M, yref, γ, λ)
#    ModelQDA{T}(order, W_k, M, priors, γ, λ)
#end
#
#doc"`discriminants(Model, Z)` Uses `Model` on input `Z` to product the class discriminants."
#function discriminants{T<:BlasReal}(mod::ModelQDA{T}, Z::Matrix{T})
#    discriminants_qda(mod.order, mod.W_k, mod.M, mod.priors, Z)
#end
#
#doc"`classify(Model, Z)` Uses `Model` on input `Z`."
#function classify{T<:BlasReal}(mod::ModelQDA{T}, Z::Matrix{T})
#    vec(mapslices(indmax, discriminants(mod, Z), mod.order == Val{:row} ? 2 : 1))
#end
