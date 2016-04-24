#==========================================================================
  Regularized Linear Discriminant Analysis Solver
==========================================================================#

immutable ModelLDA{T<:BlasReal}
    order::Union{Type{Val{:row}},Type{Val{:col}}}
    is_cda::Bool
    W::Matrix{T}       # Whitening matrix
    M::Matrix{T}       # Matrix of class means (one per row)
    priors::Vector{T}  # Vector of class priors
    gamma::Nullable{T}
end

function show(io::IO, model::ModelLDA)
    println(io, (model.is_cda ? "Canonical" : "Linear") * " Discriminant Model")
    println(io, "\nRegularization Parameters:")
    println(io, "γ = ", isnull(model.gamma) ? "n/a" : string(get(model.gamma)))
    println(io, "\nPrior Probabilities:")
    for i in eachindex(model.priors)
        println(io, "Class ", i, ": ", model.priors[i])
    end
    println(io, "\nGroup Means (one per row):")
    println(io, model.M)
end

for (scheme, dimension) in ((:(:row), 1), (:(:col), 2))
    isrowmajor = dimension == 1
    alt_dimension = isrowmajor ? 2 : 1
    ## lda!
    #whiten_qr  = isrowmajor ? :(whitendata_qr!(H)) :
    #                          :(transpose!(whitendata_qr!(transpose(H))))
    #whiten_svd = isrowmajor ? :(transpose(whitendata_svd!(H, get(γ)))) :
    #                          :(whitendata_svd!(transpose(H), get(γ)))
    ## cda!
    mu    = isrowmajor ? :(priors'M)   : :(M'priors)
    H_cda = isrowmajor ? :(Hm * W_lda) : :(W_lda * Hm)
    W_cda = isrowmajor ? :(W_lda * transpose(Vᵀ)) : :(Vᵀ * W_lda)
    @eval begin
        # X in uncentered data matrix
        # M is matrix of class means (one per row)
        # y is one-based vector of class IDs
        # λ is nullable regularization parameter in [0,1]
        function lda!{T<:BlasReal,U}(
                 ::Type{Val{$scheme}},
                X::Matrix{T},
                M::Matrix{T},
                y::RefVector{U},
                γ::Nullable{T}
            )
            H = centerclasses!(Val{$scheme}, X, M, y)
            isnull(γ) ? whitendata_qr!(Val{$scheme}, H) : whitendata_svd!(Val{$scheme}, H, get(γ))
        end

        # same rules as lda! for common arguments
        # priors is a vector of class weights
        function cda!{T<:BlasReal,U}(
                 ::Type{Val{$scheme}},
                X::Matrix{T}, 
                M::Matrix{T}, 
                y::RefVector{U}, 
                γ::Nullable{T},
                priors::Vector{T}
            )
            k = convert(Int64, y.k)
            k == length(priors) || error("Argument priors length does not match class count")
            p = size(X, $alt_dimension)
            W_lda = lda!(Val{$scheme}, X, M, y, γ)
            Hm = broadcast!(-, M, M, $mu)
            UDVᵀ = svdfact!($H_cda)
            Vᵀ = (UDVᵀ[:Vt])[1:min(k-1,p),:] # sub(UDVᵀ[:Vt], 1:min(k-1,p), :)
            $W_cda
        end
        #=
        function discriminants_lda{T<:BlasReal}(
                 ::Type{Val{$scheme}},
                W::Matrix{T},
                M::Matrix{T},
                priors::Vector{T},
                Z::Matrix{T}
            )
            n = size(Z,$dimension)
            p = size(Z,$alt_dimension)
            if size(W, $dimension) != m
                throw(DimensionMismatch("Oops"))
            end
            d = size(W, $alt_dimension)
            k = length(priors)
            δ   = Array(T, n, k) # discriminant function values
            H   = Array(T, n, p) # temporary array to prevent re-allocation k times
            #Q   = Array(T, n, d) # Q := H*W
            hᵀh = Array(T, n)    # diag(H'H)
            for j = 1:k
                broadcast!(-, H, Z, sub(M, j,:))
                dotvectors!(Val{:row}, H*W, hᵀh)
                for i = 1:n
                    δ[i, j] = -hᵀh[i]/2 + log(priors[j])
                end
            end
            δ
        end
        =#
    end
end

doc"`lda(X, y; M, gamma, priors)` Fits a regularized linear discriminant model to the data in `X` 
based on class identifier `y`."
function lda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::AbstractVector{U};
        order::Union{Type{Val{:row}},Type{Val{:col}}} = Val{:row},
        M::Matrix{T} = classmeans(order, X, RefVector(y)),
        gamma::Union{T,Nullable{T}} = Nullable{T}(),
        priors::Vector{T} = ones(T,maximum(y))/maximum(y)
    )
    all(priors .> 0) || error("Argument priors must have positive values only")
    isapprox(sum(priors), one(T)) || error("Argument priors must sum to 1")
    γ = isa(gamma, Nullable) ? gamma : Nullable(gamma)
    W = lda!(order, copy(X), M, isa(y,RefVector) ? y : RefVector(y), γ)
    ModelLDA{T}(order, false, W, M, priors, γ)
end

doc"`cda(X, y; M, gamma, priors)` Fits a regularized canonical discriminant model to the data in
`X` based on class identifier `y`."
function cda{T<:BlasReal,U<:Integer}(
        X::Matrix{T},
        y::AbstractVector{U};
        M::Matrix{T} = class_means(X,RefVector(y)),
        priors::Vector{T} = ones(T, maximum(y))/maximum(y),
        gamma::Union{T,Nullable{T}} = Nullable{T}()
    )
    all(priors .> 0) || error("Argument priors must have positive values only")
    isapprox(sum(priors), one(T)) || error("Argument priors must sum to 1")
    γ = isa(gamma, Nullable) ? gamma : Nullable(gamma)
    W = cda!(copy(X), copy(M), RefVector(y), γ, priors)
    ModelLDA{T}(true, W, M, priors, γ)
end

function discriminants_lda{T<:BlasReal}(
         ::Type{Val{:row}},
        W::Matrix{T},
        M::Matrix{T},
        priors::Vector{T},
        Z::Matrix{T}
    )
    n, p = size(Z)
    p == size(W, 1) || throw(DimensionMismatch("oops"))
    d = size(W, 2)
    k = length(priors)
    δ   = Array(T, n, k) # discriminant function values
    H   = Array(T, n, p) # temporary array to prevent re-allocation k times
    #Q   = Array(T, n, d) # Q := H*W
    hᵀh = Array(T, n)    # diag(H'H)
    for j = 1:k
        broadcast!(-, H, Z, sub(M, j,:))
        dotvectors!(Val{:row}, H*W, hᵀh)
        for i = 1:n
            δ[i, j] = -hᵀh[i]/2 + log(priors[j])
        end
    end
    δ
end

function discriminants_lda{T<:BlasReal}(
         ::Type{Val{:col}},
        W::Matrix{T},
        M::Matrix{T},
        priors::Vector{T},
        Z::Matrix{T}
    )
    n = size(Z,1)
    m = size(Z,2)
    #p == size(W, 1) || throw(DimensionMismatch("oops"))
    d = size(W, 2)
    k = length(priors)
    δ   = Array(T, n, k) # discriminant function values
    H   = Array(T, n, p) # temporary array to prevent re-allocation k times
    #Q   = Array(T, n, d) # Q := H*W
    hᵀh = Array(T, n)    # diag(H'H)
    for j = 1:k
        broadcast!(-, H, Z, sub(M, j,:))
        dotvectors!(Val{:row}, H*W, hᵀh)
        for i = 1:n
            δ[i, j] = -hᵀh[i]/2 + log(priors[j])
        end
    end
    δ
end


#=
function discriminants_lda{T<:BlasReal}(
        W::Matrix{T},
        M::Matrix{T},
        priors::Vector{T},
        Z::Matrix{T}
    )
    n, p = size(Z)
    p == size(W, 1) || throw(DimensionMismatch("oops"))
    d = size(W, 2)
    k = length(priors)
    δ   = Array(T, n, k) # discriminant function values
    H   = Array(T, n, p) # temporary array to prevent re-allocation k times
    Q   = Array(T, n, d) # Q := H*W
    hᵀh = Array(T, n)    # diag(H'H)
    for j = 1:k
        translate!(copy!(H, Z), -vec(M[j,:]))
        dotrows!(BLAS.gemm!('N', 'N', one(T), H, W, zero(T), Q), hᵀh)
        for i = 1:n
            δ[i, j] = -hᵀh[i]/2 + log(priors[j])
        end
    end
    δ
end
=#

function discriminants{T<:BlasReal}(mod::ModelLDA{T}, Z::Matrix{T})
    discriminants_lda(mod.order, mod.W, mod.M, mod.priors, Z)
end

function classify{T<:BlasReal}(mod::ModelLDA{T}, Z::Matrix{T})
    mapslices(indmax, discriminants(mod, Z), 2)
end
