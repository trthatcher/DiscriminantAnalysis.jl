#==========================================================================
  Regularized Discriminant Analysis Solvers
==========================================================================#

# Element-wise translate
function translate!{T<:AbstractFloat}(A::Array{T}, b::T)
    @inbounds for i = 1:length(A)
        A[i] += b
    end
    A
end
translate!{T<:AbstractFloat}(b::T, A::Array{T}) = translate!(A, b)

# A := A .+ b'
function translate!{T<:AbstractFloat}(b::Vector{T}, A::Matrix{T})
    (n = size(A,1)) == length(b) || throw(DimensionMismatch("first dimension of A does not match length of b"))
    @inbounds for j = 1:size(A,2), i = 1:n
        A[i,j] += b[i]
    end
    A
end

# A := b .+ A
function translate!{T<:AbstractFloat}(A::Matrix{T}, b::Vector{T})
    (n = size(A,2)) == length(b) || throw(DimensionMismatch("second dimension of A does not match length of b"))
    @inbounds for j = 1:n, i = 1:size(A,1)
        A[i,j] += b[j]
    end
    A
end

# S1 := (1-λ)S2 + λ
function regularize!{T<:AbstractFloat}(S1::Matrix{T}, λ::T, S2::Matrix{T})
    (n = size(S1,1)) == size(S1,2) || throw(DimensionMismatch("Matrix S1 must be square."))
    (m = size(S2,1)) == size(S2,2) || throw(DimensionMismatch("Matrix S2 must be square."))
    n == d || throw(DimensionMismatch("Matrix S1 and S2 must be of the same order."))
    0 <= λ <= 1 || error("λ = $(λ) must be in the interval [0,1]")
    for j = 1:n, i = 1:n
            S1[i,j] = (1-λ)*S1[i,j] + λ*S2[i,j]
    end
    S1
end

# Symmetrize the lower half of matrix S using the upper half of S
function symml!(S::Matrix)
    (p = size(S,1)) == size(S,2) || throw(ArgumentError("S must be square"))
    @inbounds for j = 1:(p - 1), i = (j + 1):p 
        S[i, j] = S[j, i]
    end
    S
end
symml(S::Matrix) = symml!(copy(S))

function sumsquare_columns{T<:AbstractFloat}(X::Matrix{T})
    n, p = size(X)
    sumsquares = zeros(n)
    for j = 1:p, i = 1:n
        sumsquares[i] += X[i,j]^2
    end
    sumsquares
end

function class_counts{T<:Integer}(y::Vector{T}, k::T = maximum(y))
    counts = zeros(Int64, k)
    for i = 1:length(y)
        y[i] <= k || error("Index $i out of range.")
        counts[y[i]] += 1
    end
    counts
end

function class_totals{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, k::U = maximum(y))
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X and y must have the same number of rows."))
    M = zeros(T, k, p)
    for j = 1:p, i = 1:n
        M[y[i],j] += X[i,j]
    end
    M
end

# Compute matrix of class means
#   X is uncentered data matrix
#   y is one-based vector of class IDs
function class_means{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, k = maximum(y))
    M = class_totals(X, y, k)
    n_k = class_counts(y, k)
    scale!(one(t) ./ n_k, M)
end


# Center rows of X based on class mean in M
#   X is uncentered data matrix
#   M is matrix of class means (one per row)
function center_rows!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::Vector{U})
    n, p = size(X)
    for j = 1:p, i = 1:n
        X[i,j] -= M[y[i],j]
    end
    X
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
        symml!(Σ_k)
    end
    Σ_k
end

# Use eigendecomposition to generate class whitening transform
#   Σ_k is array of references to each Σ_i covariance matrix
#   λ is regularization parameter in [0,1]
function class_whiteners!{T<:AbstractFloat}(Σ_k::Array{Array{T,2},1}, γ::T)
    for i = 1:length(Σ_k)
        tol = eps(T) * prod(size(Σ_k[i])) * maximum(Σ_k[i])
        Λ_i, V_i = LAPACK.syev!('V', 'U', Σ_k[i])  # Overwrite Σ_k with V such that VΛVᵀ = Σ_k
        if γ > 0
            μ_λ = mean(Λ_i)  # Shrink towards average eigenvalue
            translate!(scale!(Λ_i, 1-γ), γ*μ_λ)  # Σ = VΛVᵀ => (1-γ)Σ + γI = V((1-γ)Λ + γI)Vᵀ
        end
        all(Λ_i .>= tol) || error("Rank deficiency detected in group $i with tolerance $tol.")
        scale!(V_i, one(T) ./ sqrt(Λ_i))  # Scale V so it whitens H*V where H is centered X
    end
    Σ_k
end

# Fit regularized discriminant model
#   X in uncentered data matrix
#   M is matrix of class means (one per row)
#   y is one-based vector of class IDs
function rda!{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, M::Matrix{T}, y::Vector{U}, λ::T, γ::T, 
                                           scale_obs::Bool = true, n_k = class_counts(y))
    k = length(n_k)
    n, p = size(X)
    H = center_rows!(X, M, y)
    w_σ = one(T) ./ sqrt(sumsquare_columns(X)/n)  # scaling constant vector
    scale!(H, w_σ)
    Σ_k = class_covariances(H, y, n_k)
    if λ > 0
        Σ = H'H
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

#=
function rda{T<:AbstractFloat,U<:Integer}(X::Matrix{T}, y::Vector{U}, M::Matrix{T} = class_means(X)
                                          lambda::T = zero(T), gamma::T = zero(T))
    W_k = rda!(copy(X), M, y, 
=#



# %~%~%~%~%~%~%~%~%~% Helper Functions %~%~%~%~%~%~%~%~%

# Find group means
# Don't pass NAs in the PDA or index 0 will be accessed
#=
function classmeans{T<:FloatingPoint}(r::DaResp, X::Matrix{T})
	n,p = size(X)
	# k = length(r.counts)
	length(r.y) == n || error("X matrix and y vector column lengths do not conform")
	M = zeros(T, length(r.counts), p)
	for i = 1:n
		M[r.y.refs[i],:] += X[i, :]
	end
	M = M ./ r.counts
	return M
end
=#

# Center matrix by class mean AND scale columns
#=
function centerscalematrix{T<:FloatingPoint, U<:Integer}(X::Matrix{T}, M::Matrix{T}, index::Vector{U})
	n,p = size(X)
	Xc = Array(T,n,p)
	sd = zeros(T,1,p)
	for i = 1:n
		Xc[i,:] = X[i,:] - M[index[i],:]
		sd += Xc[i,:].^2
	end
	sd = sqrt(sd/(n-1))
	Xc = Xc ./ sd	# Scale columns
	return (Xc, vec(sd))
end
=#


# %~%~%~%~%~%~%~%~%~% Fit Methods %~%~%~%~%~%~%~%~%

# Perform regularized discriminant analysis
#=
function fitda!(dr::DaResp, dp::RdaPred{RegDiscr}; tol::Float64=0.0001)
	nk = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centerscalematrix(dp.X,dp.means,dr.y.refs)
	Sigma = Xc' * Xc	# Compute Cov(X) for lambda regularization
	Sigma_k = Array(Float64,p,p)
	for k = 1:nk	# BLAS/LAPACK optimizations?
		class_k = find(dr.y.refs .== k)
		Sigma_k = (Xc[class_k,:])' * Xc[class_k,:]
		Sigma_k = (1-dp.discr.lambda) * Sigma_k + dp.discr.lambda * Sigma	# Shrink towards Pooled covariance
		s, V = svd(Sigma_k,false)[2:3]
		s = s ./ ((1-dp.discr.lambda)*(dr.counts[k]-1) + dp.discr.lambda*(n-nk))
		if dp.discr.gamma != 0
			s = s .* (1-dp.discr.gamma) .+ (dp.discr.gamma * trace(Sigma_k) / p)	# Shrink towards (I * Average Eigenvalue)
		end
		rank = sum(s .> maximum(s)*tol)
		rank == p || error("Rank deficiency detected in group $k with tolerance $tol")
		dp.discr.whiten[:,:,k] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))	# Whitening transformation for class k
	end
end
=#

# Perform quadratic discriminant analsis
#=
function fitda!(dr::DaResp, dp::RdaPred{QuadDiscr}; tol::Float64=0.0001)
	nk = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centerscalematrix(dp.X,dp.means,dr.y.refs)
	for k = 1:nk
		class_k = find(dr.y.refs .==k)
		s, V = svd(Xc[class_k,:],false)[2:3]
		if length(s) < p s =  vcat(s, zeros(p - length(s))) end
		if dp.discr.gamma != 0	# Shrink towards (I * Average Eigenvalue)
			s = (s .^ 2)/(dr.counts[k]-1) .* (1-dp.discr.gamma) .+ (dp.discr.gamma * sum(s) / p)
		else	# No shrinkage
			s = (s .^ 2)/(dr.counts[k]-1)
		end
		rank = sum(s .> s[1]*tol)
		rank == p || error("Rank deficiency detected in group $k with tolerance=$tol.")
		dp.discr.whiten[:,:,k] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
	end
end
=#

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

function pred(dp::Union(RdaPred{RegDiscr},RdaPred{QuadDiscr}), X::Matrix{Float64})
	n,p = size(X)
	ng = length(dp.logpr)
	Zk = Array(Float64,n,p)
	P = Array(Float64, n, ng)
	for k=1:ng
		Zk = (X .- dp.means[k,:]) * dp.discr.whiten[:,:,k]
		P[:,k] = mapslices(x -> -0.5*sum(x .^ 2), Zk, 2) .+ dp.logpr[k]
	end
	return mapslices(indmax,P,2)
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
