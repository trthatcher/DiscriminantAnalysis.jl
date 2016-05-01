#==========================================================================
  Regularized Quadratic Discriminant Analysis Solver
==========================================================================#

immutable ModelQDA{T<:BlasReal}
    order::Union{Type{Val{:row}},Type{Val{:col}}}
    W_k::Vector{Matrix{T}}  # Vector of class whitening matrices
    M::Matrix{T}            # Matrix of class means (one per row)
    priors::Vector{T}       # Vector of class priors
    gamma::Nullable{T}
    lambda::Nullable{T}
end

function show(io::IO, model::ModelQDA)
    println(io, "Quadratic Discriminant Model")
    println(io, "\nRegularization Parameters:")
    print(io, "    γ = ")
    isnull(model.gamma) ? print(io, "N/A") : showcompact(get(model.gamma))
    print(io, "    λ = ")
    isnull(model.lambda) ? print(io, "N/A") : showcompact(get(model.lambda))
    println(io, "\n\nClass Priors:")
    for i in eachindex(model.priors)
        print(io, "    Class ", i, " Probability: ")
        showcompact(io, model.priors[i]*100)
        println(io, "%")
    end
    println(io, "\nClass Means:")
    for j in eachindex(model.priors)
        print(io, "    Class ", j, " Mean: [")
        print(io, join([sprint(showcompact, v) for v in subvector(model.order, model.M, j)], ", "))
        println(io, "]")
    end
    print(io, "\n")
end

function covmatrix{T<:BlasReal}(::Type{Val{:row}}, H::Matrix{T})
    n = size(H,1)
    Σ = H'H
    broadcast!(/, Σ, Σ, n-1)
end

function covmatrix{T<:BlasReal}(::Type{Val{:col}}, H::Matrix{T})
    n = size(H,2)
    Σ = A_mul_Bt(H, H)
    broadcast!(/, Σ, Σ, n-1)
end

for (scheme, dim_obs) in ((:(:row), 1), (:(:col), 2))
    isrowmajor = dim_obs == 1
    dim_param = isrowmajor ? 2 : 1

    H_i      = isrowmajor ? :(H[y .== i, :]) : :(H[:, y .== i])
    _ij, _ji = isrowmajor ? (:i, :j) : (:j, :i)  # Swapped variables for row and column ordering
    _nk, _kn = isrowmajor ? (:n, :k) : (:k, :n)
    _np, _pn = isrowmajor ? (:n, :p) : (:p, :n)
    W_j, H   = isrowmajor ? (:(W_k[j]), :H) : (:H, :(W_k[j]))

    @eval begin
        # λ-regularized QDA - require full covariance matrices
        function classwhiteners!{T<:BlasReal,U}(
                 ::Type{Val{$scheme}},
                H::Matrix{T},
                y::RefVector{U}, 
                γ::Nullable{T},
                λ::T
            )
            k = convert(Int64, y.k)
            Σ_k = AbstractMatrix{T}[covmatrix(Val{$scheme}, $H_i) for i = 1:k]
            Σ   = covmatrix(Val{$scheme}, H)
            for i in eachindex(Σ_k)
                regularize!(Σ_k[i], λ, Σ)
                Σ_k[i] = whitencov_chol!(Val{$scheme}, Σ_k[i], γ)
            end
            Σ_k
        end

        # No λ-regularization - only need data matrices
        function classwhiteners!{T<:BlasReal,U}(
                 ::Type{Val{$scheme}},
                H::Matrix{T},
                y::RefVector{U},
                γ::Nullable{T}
            )
            k = convert(Int64, y.k)
            if isnull(γ)
                AbstractMatrix{T}[whitendata_qr!(Val{$scheme}, $H_i) for i = 1:k]
            else
                AbstractMatrix{T}[whitendata_svd!(Val{$scheme}, $H_i, get(γ)) for i = 1:k]
            end
        end

        function qda!{T<:BlasReal,U}(
                 ::Type{Val{$scheme}},
                X::Matrix{T}, 
                M::Matrix{T}, 
                y::RefVector{U}, 
                γ::Nullable{T},
                λ::Nullable{T}
            )
            H = centerclasses!(Val{$scheme}, X, M, y)
            if isnull(λ)
                classwhiteners!(Val{$scheme}, H, y, γ)
            else
                classwhiteners!(Val{$scheme}, H, y, γ, get(λ))
            end
        end

        function discriminants_qda{T<:BlasReal}(
                 ::Type{Val{$scheme}},
                W_k::Vector{Matrix{T}},
                M::Matrix{T},
                priors::Vector{T},
                Z::Matrix{T}
            )
            n = size(Z, $dim_obs)
            p = size(Z, $dim_param)
            k = length(priors)
            D   = Array(T, $_nk, $_kn)  # discriminant function values
            H   = Array(T, $_np, $_pn)  # temporary array to prevent re-allocation k times
            hᵀh = Array(T, n)           # diag(H'H)
            for j = 1:k
                broadcast!(-, H, Z, subvector(Val{$scheme}, M, j))
                dotvectors!(Val{$scheme}, $H * $W_j, hᵀh)
                for i = 1:n
                    D[$_ij, $_ji] = -hᵀh[i]/2 + log(priors[j])
                end
            end
            D
        end
    end
end

doc"`qda(X, y; order, M, lambda, gamma, priors)` Fits a regularized quadratic discriminant model to
the data in `X` based on class identifier `y`."
function qda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::AbstractVector{U};
        order::Union{Type{Val{:row}},Type{Val{:col}}} = Val{:row},
        M::Matrix{T} = classmeans(order, X, RefVector(y)),
        gamma::Union{T,Nullable{T}}  = Nullable{T}(),
        lambda::Union{T,Nullable{T}} = Nullable{T}(),
        priors::Vector{T} = ones(T,maximum(y))/maximum(y)
    )
    all(priors .> 0) || error("Argument priors must have positive values only")
    isapprox(sum(priors), one(T)) || error("Argument priors must sum to 1")
    γ = isa(gamma,  Nullable) ? gamma  : Nullable(gamma)
    λ = isa(lambda, Nullable) ? lambda : Nullable(lambda)
    W_k = qda!(order, copy(X), M, isa(y,RefVector) ? y : RefVector(y), γ, λ)
    ModelQDA{T}(order, W_k, M, priors, γ, λ)
end

doc"`discriminants(Model, Z)` Uses `Model` on input `Z` to product the class discriminants."
function discriminants{T<:BlasReal}(mod::ModelQDA{T}, Z::Matrix{T})
    discriminants_qda(mod.order, mod.W_k, mod.M, mod.priors, Z)
end

doc"`classify(Model, Z)` Uses `Model` on input `Z`."
function classify{T<:BlasReal}(mod::ModelQDA{T}, Z::Matrix{T})
    mapslices(indmax, discriminants(mod, Z), mod.order == Val{:row} ? 2 : 1)
end
