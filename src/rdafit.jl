# %~%~%~%~%~%~%~%~%~% Helper Functions %~%~%~%~%~%~%~%~%

# Find group means
# Don't pass NAs in the PDA or index 0 will be accessed
function groupmeans{T<:FP}(r::DaResp, X::Matrix{T})
	n,p = size(X)
	k = length(r.counts)
	length(r.y) == n || error("Array lengths do not conform")
	M = zeros(T, k, p)
	for i = 1:n
		M[r.y.refs[i],:] += X[i, :]
	end
	M = M ./ r.counts
	return M
end

# Center matrix by group mean AND scale columns
function centerscalematrix{T<:FP, U<:Integer}(X::Matrix{T}, M::Matrix{T}, index::Vector{U})
	n,p = size(X)
	Xc = Array(Float64,n,p)
	sd = zeros(Float64,1,p)
	for i = 1:n
		Xc[i,:] = X[i,:] - M[index[i],:]
		sd += Xc[i,:].^2
	end
	sd = sqrt(sd/(n-1))	# Unbiased estiate of the sd
	Xc = Xc ./ sd
	return (Xc, vec(sd))
end


# %~%~%~%~%~%~%~%~%~% Fit Function %~%~%~%~%~%~%~%~%


# Rework to have response
function fitda!(dr::DaResp, dp::RdaPred{RegDiscr})
	ng = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centerscalematrix(dp.X,dp.means,dr.y.refs)
	Sigma = Xc' * Xc	# Compute Cov(X) for lambda regularization
	Sigma_k = Array(Float64,p,p)
	for k = 1:ng	# BLAS/LAPACK optimizations?
		class_k = find(dr.y.refs .== k)
		Sigma_k = (Xc[class_k,:])' * Xc[class_k,:]
		Sigma_k = (1-dp.discr.lambda) * Sigma_k + dp.discr.lambda * Sigma		# Shrink towards Pooled covariance
		EFact = eigfact(Sigma_k)
			s = EFact[:values]
			V = EFact[:vectors]
		s = s ./ ((1-dp.discr.lambda)*(dr.counts[k]-1) + dp.discr.lambda*(n-ng))
		if dp.discr.gamma != 0
			s = s .* (1-dp.discr.gamma) .+ (dp.discr.gamma * trace(Sigma_k) / p)	# Shrink towards (I * Average Eigenvalue)
		end
		dp.discr.whiten[:,:,k] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))	# Whitening transformation for group k
	end
end

function fitda!(dr::DaResp, dp::RdaPred{QuadDiscr})
	ng = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centerscalematrix(dp.X,dp.means,dr.y.refs)
	for k = 1:ng
		class_k = find(dr.y.refs .==k)
		s, V = svd(Xc[class_k,:])[2:3]
		if dp.discr.gamma != 0 
			# Shrink towards (I * Average Eigenvalue)
			s = (s .^ 2)/(dr.counts[k]-1) .* (1-dp.discr.gamma) .+ (dp.discr.gamma * sum(s) / p)
		else	# No shrinkage
			s = (s .^ 2)/(dr.counts[k]-1)
		end
		dp.discr.whiten[:,:,k] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
	end
end

function fitda!(dr::DaResp, dp::RdaPred{LinDiscr})
	println("Linear Discriminant Analysis")
	ng = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centerscalematrix(dp.X,dp.means,dr.y.refs)
	s, V = svd(Xc)[2:3]
	if dp.discr.gamma != 0 
		# Shrink towards (I * Average Eigenvalue)
		s = (s .^ 2)/(n-ng) .* (1-dp.discr.gamma) .+ (dp.discr.gamma * sum(s) / p)
	else	# No shrinkage
		s = (s .^ 2)/(n-ng)	# Check division properly
	end
	dp.discr.whiten[:,:] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
	if (rrlda == true) & (ng > 2)
		tol = 0.0001
		mu = sum(dr.priors .* dp.means, 1)
		Mc = (dp.means .- mu) * dp.discr.whiten[:,:]
		s, V = svd(Mc)[2:3]
		rank = sum(s .> s[0]*tol)
		print("Rank is: ")
		println(rank)
		dp.discr.whiten = dp.discr.whiten * V[:,1:rank]
	end
end



# %~%~%~%~%~%~%~%~%~% Wrapper Functions %~%~%~%~%~%~%~%~%


function rda(f::Formula, df::AbstractDataFrame; priors::Vector{Float64}=Float64[], lambda=0.5, gamma=0, rrlda=true)
	mf::ModelFrame = ModelFrame(f, df)
	X::Matrix{Float64} = ModelMatrix(mf).m[:,2:]
	n, p = size(X)
	y::PooledDataVector = PooledDataArray(model_response(mf)) 	# NOTE: pooled conversion done INSIDE rda in case df leaves out factor levels
	k = length(levels(y))
	lp = length(priors)
		lp == 0 || lp == k || error("length(priors) = $lp should be 0 or $k")
	pr = lp == k ? copy(priors) : ones(Float64, k)/k
	dr = DaResp{Float64}(y, pr)
	if lambda == 1
		dp = RdaPred{LinDiscr}(X, groupmeans(dr,X), LinDiscr(Array(Float64,p,p), gamma, rrlda), log(pr))
	elseif lambda == 0
		discr = QuadDiscr(Array(Float64,p,p,k), gamma)
		dp = RdaPred{QuadDiscr}(X, groupmeans(dr,X), discr, log(pr))
	else
		discr = RegDiscr(Array(Float64,p,p,k), lambda, gamma)
		dp = RdaPred{RegDiscr}(X, groupmeans(dr,X), discr, log(pr))
	end
	fitda!(dr,dp)
	return DaModel(mf,dr,dp,f)
end

lda(f, df; priors=Float64[], gamma=0, rrlda=true) = rda(f, df; priors=priors, lambda=1, gamma=gamma, rrlda=rrlda)
qda(f, df; priors=Float64[], gamma=0) = rda(f, df; priors=priors, lambda=0, gamma=gamma)

# %~%~%~%~%~%~%~%~%~% Wrapper Functions %~%~%~%~%~%~%~%~%

function predict(mod::DaModel, X::Matrix{Float64})
	return pred(mod.dp,X)
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


