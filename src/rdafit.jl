# %~%~%~%~%~%~%~%~%~% Helper Functions %~%~%~%~%~%~%~%~%

# Find group means
# Don't pass NAs in the PDA or index 0 will be accessed
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

# Center matrix by class mean AND scale columns
function centerscalematrix{T<:FloatingPoint, U<:Integer}(X::Matrix{T}, M::Matrix{T}, index::Vector{U})
	n,p = size(X)
	Xc = Array(Float64,n,p)
	sd = zeros(Float64,1,p)
	for i = 1:n
		Xc[i,:] = X[i,:] - M[index[i],:]
		sd += Xc[i,:].^2
	end
	sd = sqrt(sd/(n-1))
	Xc = Xc ./ sd	# Scale columns
	return (Xc, vec(sd))
end



# %~%~%~%~%~%~%~%~%~% Fit Methods %~%~%~%~%~%~%~%~%

# Perform regularized discriminant analysis
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

# Perform quadratic discriminant analsis
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

# Perform linear discriminant analysis (rank-reduced is default)
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



# %~%~%~%~%~%~%~%~%~% Frontend %~%~%~%~%~%~%~%~%

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



# %~%~%~%~%~%~%~%~%~% Object Access %~%~%~%~%~%~%~%~%

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


# %~%~%~%~%~%~%~%~%~% Prediction Methods %~%~%~%~%~%~%~%~%

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


