#==========================================================================
  Regularized Discriminant Analysis Solvers
==========================================================================#

immutable ModelQDA{T<:AbstractFloat}
    W_k::Array{Matrix{T},1}  # Vector of class whitening matrices
    M::Matrix{T}             # Matrix of class means (one per row)
    priors::Vector{T}        # Vector of class priors
end

# Create an array of class scatter matrices
#   H is centered data matrix (with respect to class means)
#   y is one-based vector of class IDs
function class_covariances{T<:AbstractFloat,U<:Integer}(H::Matrix{T}, y::Vector{U}, 
                                                        n_k::Vector{Int64} = class_counts(y))
    k = length(n_k)
    p = size(H,2)
    Σ_k = Array(Array{T,2}, k)
    for i = 1:k  # Σ_k[i] = H_i'H_i/(n_i-1)
        Σ_k[i] = BLAS.syrk!('U', 'T', one(T)/(n_k[i]-1), H[y .== i,:], zero(T), Array(T,p,p))
        symml!(Σ_k[i])
    end
    Σ_k
end

# Use eigendecomposition to generate class whitening transform
#   Σ_k is array of references to each Σ_i covariance matrix
#   λ is regularization parameter in [0,1]
function class_whiteners!{T<:AbstractFloat}(Σ_k::Vector{Matrix{T}}, γ::T)
    for i = 1:length(Σ_k)
        tol = eps(T) * prod(size(Σ_k[i])) * maximum(Σ_k[i])
        Λ_i, V_i = LAPACK.syev!('V', 'U', Σ_k[i])  # Overwrite Σ_k with V such that VΛVᵀ = Σ_k
        if γ > 0
            μ_λ = mean(Λ_i)  # Shrink towards average eigenvalue
            translate!(scale!(Λ_i, 1-γ), γ*μ_λ)  # Σ = VΛVᵀ => (1-γ)Σ + γI = V((1-γ)Λ + γI)Vᵀ
        end
        all(Λ_i .>= tol) || error("Rank deficiency detected in class $i with tolerance $tol.")
        scale!(V_i, one(T) ./ sqrt(Λ_i))  # Scale V so it whitens H*V where H is centered X
    end
    Σ_k
end

# Fit regularized discriminant model
#   X in uncentered data matrix
#   M is matrix of class means (one per row)
#   y is one-based vector of class IDs
function qda!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::Vector{U}, λ::T, γ::T, 
                                           n_k = class_counts(y))
    k = length(n_k)
    n, p = size(X)
    H = center_rows!(X, M, y)
    w_σ = one(T) ./ sqrt(dot_columns(X)/n)  # scaling constant vector
    scale!(H, w_σ)
    Σ_k = class_covariances(H, y, n_k)
    if λ > 0
        Σ = scale!(H'H, one(T)/(n-1))
        for i = 1:k 
            regularize!(Σ_k[i], λ, Σ)
        end
    end
    W_k = class_whiteners!(Σ_k, γ)
    for i = 1:k
        scale!(W_k[i], w_σ) 
    end
    W_k
end

function qda{T<:AbstractFloat,U<:Integer}(
        X::Matrix{T},
        y::Vector{U};
        M::Matrix{T} = class_means(X,y),
        lambda::T = zero(T),
        gamma::T = zero(T),
        priors::Vector{T} = T[1/maximum(y) for i = 1:maximum(y)]
    )
    n_k = class_counts(y)
    W_k = qda!(copy(X), M, y, lambda, gamma, n_k)
    ModelQDA{T}(W_k, M, priors)
end

function predict_qda{T<:AbstractFloat}(W_k::Vector{Matrix{T}}, M::Matrix{T}, priors::Vector{T},
                                       Z::Matrix{T})
    n, p = size(Z)
    k = length(W_k)
    size(M,2) == p || throw(DimensionMismatch("Z does not have the same number of columns as M."))
    size(M,1) == k || error("class mismatch")
    length(priors) == k || error("class mismatch")
    δ = Array(T, n, k)      # discriminant function values
    Z_tmp = Array(T, n, p)  # temporary array to prevent re-allocation k times
    Z_j = Array(T, n, p)    # Z in W_k
    for j = 1:k
        translate!(copy!(Z_tmp, Z), vec(M[j,:]))
        BLAS.gemm!('N', 'N', one(T), Z_tmp, W_k[j], zero(T), Z_j)
        s = dot_rows(Z_j)
        for i = 1:n
            δ[i, j] = -s[i]/2 + log(priors[j])
        end
    end
    mapslices(indmax, δ, 2)
end

function predict{T<:AbstractFloat}(mod::ModelQDA{T}, Z::Matrix{T})
    predict_qda(mod.W_k, mod.M, mod.priors, Z)
end


# %~%~%~%~%~%~%~%~%~% Helper Functions %~%~%~%~%~%~%~%~%

# Perform linear discriminant analysis (rank-reduced is default)
#=
function fitda!(dr::DaResp, dp::RdaPred{LinDiscr}; tol::Float64=0.0001)
	nk = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centerscalematrix(dp.X,dp.means,dr.y.refs)
	s, V = svd(Xc,false)[2:3]
	if length(s) < p s =  vcat(s, zeros(p - length(s))) end
	if dp.discr.gamma != 0 	# Shrink towards (I * Average Eigenvalue)
		s = (s .^ 2)/(n-nk) .* (1-dp.discr.gamma) .+ (dp.discr.gamma * sum(s) / p)
	else	# No shrinkage
		s = (s .^ 2)/(n-nk)
	end
	rank = sum(s .> s[1]*tol)
	rank == p || error("Rank deficiency detected with tolerance=$tol.")
	dp.discr.whiten = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
	if (dp.discr.rrlda == true) & (nk > 2)
		mu = sum(dr.priors .* dp.means, 1)
		Mc = (dp.means .- mu) * dp.discr.whiten
		s, V = svd(Mc, false)[2:3]
		rank = sum(s .> s[1]*tol)
		dp.discr.whiten = dp.discr.whiten * V[:,1:rank]
	end
end
=#


# %~%~%~%~%~%~%~%~%~% Frontend %~%~%~%~%~%~%~%~%

#=
function rda(f::Formula, df::AbstractDataFrame; priors::Vector{Float64}=Float64[], lambda::Real=0.5, gamma::Real=0, rrlda::Bool=true, tol::Real=0.0001)
	(lambda >= 0) & (lambda <= 1) || error("Lambda=$lambda should be between 0 and 1 inclusive")
	(gamma >= 0) & (gamma <= 1) || error("Gamma=$gamma should be between 0 and 1 inclusive")
	(tol >= 0) & (tol <= 1) || error("Tolerance=$tol should be between 0 and 1 inclusive")
	mf = ModelFrame(f, df)
	X = ModelMatrix(mf).m[:,2:]
	n, p = size(X)
	y = PooledDataArray(model_response(mf)) 	# NOTE: pdv conversion done INSIDE rda in case df leaves out factor levels
	k = length(levels(y))
	lp = length(priors)
		lp == 0 || lp == k || error("length(priors) = $lp should be 0 or $k")
	pr = lp == k ? copy(priors) : ones(Float64, k)/k
	dr = DaResp(y, pr)
	if lambda == 1
		dp = RdaPred{LinDiscr}(X, classmeans(dr,X), LinDiscr(Array(Float64,p,p), gamma, rrlda), log(pr))
	elseif lambda == 0
		discr = QuadDiscr(Array(Float64,p,p,k), gamma)
		dp = RdaPred{QuadDiscr}(X, classmeans(dr,X), discr, log(pr))
	else
		discr = RegDiscr(Array(Float64,p,p,k), lambda, gamma)
		dp = RdaPred{RegDiscr}(X, classmeans(dr,X), discr, log(pr))
	end
	fitda!(dr,dp,tol=tol)
	return DaModel(mf,dr,dp,f)
end

lda(f, df; priors=Float64[], gamma=0, rrlda=true, tol=0.0001) = rda(f, df; priors=priors, lambda=1, gamma=gamma, rrlda=rrlda, tol=tol)
qda(f, df; priors=Float64[], gamma=0, tol=0.0001) = rda(f, df; priors=priors, lambda=0, gamma=gamma, tol=tol)
=#


# %~%~%~%~%~%~%~%~%~% Object Access %~%~%~%~%~%~%~%~%
#=
classes(mod::DaModel) = levels(mod.dr.y)
priors(mod::DaModel) = mod.dr.priors
counts(mod::DaModel) = mod.dr.counts
formula(mod::DaModel) = mod.f
scaling(mod::DaModel) = mod.dp.discr.whiten
whiten(mod::DaModel) = mod.dp.discr.whiten
means(mod::DaModel) = mod.dp.means
logpriors(mod::DaModel) = mod.dp.logpr
gamma(mod::DaModel) = mod.dp.discr.gamma
lambda(mod::DaModel) = lambda(mod.dp.discr)
	lambda(d::RegDiscr) = d.lambda
	lambda(d::QuadDiscr) = 0
	lambda(d::LinDiscr) = 1
rankreduced(mod::DaModel) = mod.dp::RdaPred{LinDiscr} ? mod.dp.discr.rrlda : false
=#

# %~%~%~%~%~%~%~%~%~% Prediction Methods %~%~%~%~%~%~%~%~%
#=
function predict(mod::DaModel, X::Matrix{Float64})
	D = index_to_level(mod.dr.y)
	return PooledDataArray(map(x->get(D,convert(Uint32,x),0), pred(mod.dp, X)))
end

function predict(mod::DaModel, df::AbstractDataFrame)
	X = ModelMatrix(ModelFrame(mod.f, df)).m[:,2:]
	return predict(mod, X)
end

function pred(dp::RdaPred{LinDiscr}, X::Matrix{Float64})
	n,p = size(X)
	ng = length(dp.logpr)
	Zk = Array(Float64,n,p)
	P = Array(Float64, n, ng)
	for k=1:ng
		Zk = (X .- dp.means[k,:]) * dp.discr.whiten
		P[:,k] = mapslices(x -> -0.5*sum(x .^ 2), Zk, 2) .+ dp.logpr[k]
	end
	return mapslices(indmax,P,2)
end
=#
