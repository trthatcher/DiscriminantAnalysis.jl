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
    print(io, "    γ = ")
    isnull(model.gamma) ? print(io, "N/A") : showcompact(get(model.gamma))
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

# Extracts Fisher components
function components!{T<:BlasReal}(
         ::Type{Val{:row}},
        W::AbstractMatrix{T},
        M::Matrix{T},
        w::Vector{T}
    )
    H = broadcast!(-, M, M, w'M)
    UDVᵀ = svdfact!(H * W)
    d = min(size(M,2),size(M,1)-1)
    Vᵀ = sub(UDVᵀ[:Vt], 1:d, :) # (UDVᵀ[:Vt])[1:d,:]
    W * transpose(Vᵀ)
end

function components!{T<:BlasReal}(
         ::Type{Val{:col}},
        W::AbstractMatrix{T},
        M::Matrix{T}, 
        w::Vector{T}
    )
    H = broadcast!(-, M, M, M*w)
    UDVᵀ = svdfact!(transpose(W * H))
    d = min(size(M,1), size(M,2)-1)
    Vᵀ = sub(UDVᵀ[:Vt], 1:d, :) #(UDVᵀ[:Vt])[1:d,:]
    Vᵀ * W
end

for (scheme, dim_obs) in ((:(:row), 1), (:(:col), 2))
    isrowmajor = dim_obs == 1
    dim_param = isrowmajor ? 2 : 1

    D_i, D_j = isrowmajor ? (:i, :j) : (:j, :i)
    D_n, D_k = isrowmajor ? (:n, :k) : (:k, :n)
    H_n, H_p = isrowmajor ? (:n, :p) : (:p, :n)
    W, H     = isrowmajor ? (:W, :H) : (:H, :W)
    
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
            p = size(X, $dim_param)
            W = lda!(Val{$scheme}, X, M, y, γ)
            components!(Val{$scheme}, W, M, priors)
        end

        function discriminants_lda{T<:BlasReal}(
                 ::Type{Val{$scheme}},
                W::Matrix{T},
                M::Matrix{T},
                priors::Vector{T},
                Z::Matrix{T}
            )
            n = size(Z, $dim_obs)
            p = size(Z, $dim_param)
            if p != size(W, $dim_param)
                throw(DimensionMismatch("Parameters of W and Z should match"))
            end
            d = size(W, $dim_param)
            k = length(priors)
            D   = Array(T, $D_n, $D_k) # discriminant function values
            H   = Array(T, $H_n, $H_p) # temporary array to prevent re-allocation k times
            hᵀh = Array(T, n)          # diag(H'H)
            for j = 1:k
                broadcast!(-, H, Z, subvector(Val{$scheme}, M, j))
                dotvectors!(Val{$scheme}, $H * $W, hᵀh)
                for i = 1:n
                    D[$D_i, $D_j] = -hᵀh[i]/2 + log(priors[j])
                end
            end
            D
        end
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
        order::Union{Type{Val{:row}},Type{Val{:col}}} = Val{:row},
        M::Matrix{T} = classmeans(order, X, RefVector(y)),
        gamma::Union{T,Nullable{T}} = Nullable{T}(),
        priors::Vector{T} = ones(T, maximum(y))/maximum(y)
    )
    all(priors .> 0) || error("Argument priors must have positive values only")
    isapprox(sum(priors), one(T)) || error("Argument priors must sum to 1")
    γ = isa(gamma, Nullable) ? gamma : Nullable(gamma)
    W = cda!(order, copy(X), copy(M), isa(y,RefVector) ? y : RefVector(y), γ, priors)
    ModelLDA{T}(order, true, W, M, priors, γ)
end

function discriminants{T<:BlasReal}(mod::ModelLDA{T}, Z::Matrix{T})
    discriminants_lda(mod.order, mod.W, mod.M, mod.priors, Z)
end

function classify{T<:BlasReal}(mod::ModelLDA{T}, Z::Matrix{T})
    mapslices(indmax, discriminants(mod, Z), 2)
end
