#==========================================================================
  Regularized Quadratic Discriminant Analysis Solver
==========================================================================#

immutable ModelQDA{T<:BlasReal}
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

gramian{T<:BlasReal}(::Type{Val{:row}}, X::Matrix{T}) = X'X
gramian{T<:BlasReal}(::Type{Val{:col}}, X::Matrix{T}) = At_mul_B(X,X)

# λ-regularized QDA - require full covariance matrices
function classwhiteners!{T<:BlasReal,U}(
         ::Type{Val{:row}},
        H::Matrix{T},
        y::RefVector{U}, 
        γ::Nullable{T},
        λ::T
    )
    k = convert(Int64, y.k)
    f_k = sqrt(one(T) ./ (classcounts(y) .- one(T)))
    Σ_k = Matrix{T}[scale!(gramian(Val{:row}, H[y .== i,:]), f_k[i]) for i = 1:k]
    Σ   = scale!(gramian(Val{:row}, H), sqrt(one(T)/(size(H,1)-1)))
    for S in Σ_k
        regularize!(S, λ, Σ)
        whitencov_chol!(Val{:row}, S, γ)
    end
    Σ_k
end

# No λ-regularization - only need data matrices
function classwhiteners!{T<:BlasReal,U}(
         ::Type{Val{:row}},
        H::Matrix{T},
        y::RefVector{U},
        γ::Nullable{T}
    )
    k = convert(Int64, y.k)
    if isnull(γ)
        Matrix{T}[whitendata_qr!(Val{:row}, H[y .== i,:]) for i = 1:k]
    else
        Matrix{T}[whitendata_svd!(Val{:row}, H[y .== i, :], get(γ)) for i = 1:k]
    end
end

function qda!{T<:BlasReal,U}(
         ::Type{Val{:row}},
        X::Matrix{T}, 
        M::Matrix{T}, 
        y::RefVector{U}, 
        γ::Nullable{T},
        λ::Nullable{T}
    )
    H = center_classes!(X, M, y)
    if isnull(λ)
        class_whiteners!(Val{:row}, H, y, γ)
    else
        class_whiteners!(Val{:row}, H, y, γ, get(λ))
    end
end

doc"`qda(X, y; M, lambda, gamma, priors)` Fits a regularized quadratic discriminant model to the 
data in `X` based on class identifier `y`."
function qda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::AbstractVector{U};
        M::Matrix{T} = class_means(X,RefVector(y)),
        gamma::Union{T,Nullable{T}}  = Nullable{T}(),
        lambda::Union{T,Nullable{T}} = Nullable{T}(),
        priors::Vector{T} = ones(T,maximum(y))/maximum(y)
    )
    all(priors .> 0) || error("Argument priors must have positive values only")
    isapprox(sum(priors), one(T)) || error("Argument priors must sum to 1")
    γ = isa(gamma,  Nullable) ? deepcopy(gamma)  : Nullable(gamma)
    λ = isa(lambda, Nullable) ? deepcopy(lambda) : Nullable(lambda)
    W_k = qda!(copy(X), M, isa(y,RefVector) ? y : RefVector(y), γ, λ)
    ModelQDA{T}(W_k, M, priors, γ, λ)
end

function discriminants_qda{T<:BlasReal}(
        W_k::Vector{Matrix{T}},
        M::Matrix{T},
        priors::Vector{T},
        Z::Matrix{T}
    )
    n, p = size(Z)
    k = length(priors)
    size(M,2) == p || throw(DimensionMismatch("Z does not have the same number of columns as M."))
    size(M,1) == k || error("class mismatch")
    length(W_k) == k || error("class mismatch")
    δ = Array(T, n, k)  # discriminant function values
    H = Array(T, n, p)  # temporary array to prevent re-allocation k times
    Q = Array(T, n, p)  # Q := H*W_k
    hᵀh = Array(T, n)
    for j = 1:k
        translate!(copy!(H, Z), -vec(M[j,:]))
        dotrows!(BLAS.gemm!('N', 'N', one(T), H, W_k[j], zero(T), Q), hᵀh)
        for i = 1:n
            δ[i, j] = -hᵀh[i]/2 + log(priors[j])
        end
    end
    δ
end

doc"`discriminants(Model, Z)` Uses `Model` on input `Z` to product the class discriminants."
function discriminants{T<:BlasReal}(mod::ModelQDA{T}, Z::Matrix{T})
    discriminants_qda(mod.W_k, mod.M, mod.priors, Z)
end

doc"`classify(Model, Z)` Uses `Model` on input `Z`."
function classify{T<:BlasReal}(mod::ModelQDA{T}, Z::Matrix{T})
    mapslices(indmax, discriminants(mod, Z), 2)
end
