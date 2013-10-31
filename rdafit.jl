type DaResp{T<:FP} <: ModResp
	y::PooledDataArray	# Response vector
	priors::Vector{Float64}	# Prior weights
	counts::Vector{Int64}	# Prior observation counts
	function DaResp(y::PooledDataArray, priors::Vector{T})
		k = length(priors)
		k == length(levels(y)) || error("Length mismatch priors/levels")
		n = length(y)
		c = zeros(Int64,k)
		for i = 1:n
			c[y.refs[i]] += 1
		end
		new(y, priors, c)
	end
end

abstract Discr

type RegDiscr <: Discr
	whiten::Array{Float64,3}
	lambda::Float64
	gamma::Real
	coef::Matrix{Float64}
	intercept::Vector{Float64}
end

type LinDiscr <: Discr
	whiten::Matrix{Float64}
	gamma::Real
end

type QuadDiscr <: Discr
	whiten::Array{Float64,3}
	gamma::Real
	coef::Matrix{Float64}
	intercept::Vector{Float64}
end

type RdaPred{T<:Discr} <: DaPred
	X::Matrix{Float64}
	means::Matrix{Float64}
	discr::T
	logpr::Vector{Float64}
end

type DaModel
	mf::ModelFrame
	dr::DaResp
	dp::DaPred
	f::Formula
end


# %~%~%~%~%~%~%~%~%~% Helper Functions %~%~%~%~%~%~%~%~%

# Find group means
# Don't pass NAs in the PDA or index 0 will be accessed
function groupmeans{T<:FP}(r::DaResp, X::Matrix{T})
	n,p = size(X)
	k = length(r.counts)
	length(r.y) == n || error("Array lengths do not conform")
	g = zeros(T, k, p)
	for i = 1:n
		g[r.y.refs[i],:] += X[r.y.refs[i], :]
	end
	for i = 1:k
		g[i,:] = g[i,:] / r.counts[i]
	end
	return g
end

# Center matrix by group mean
function centermatrix{T<:FP, U<:Integer}(X::Matrix{T}, M::Matrix{T}, index::Vector{U})
	n,p = size(X)
	Xc = Array(Float64,n,p)
	sd = zeros(Float64,1,p)
	for i = 1:n
		Xc[i,:] = X[i,:] - M[index[i],:]
		sd += Xc[i,:].^2
	end
	sd = sqrt(sd/(n-1))	# Unbiased estiate of the sd
	for i = 1:n	# BLAS improvement?
		Xc[i,:] = Xc[i,:] ./ sd
	end
	return (Xc, vec(sd))
end


# %~%~%~%~%~%~%~%~%~% Fit Function %~%~%~%~%~%~%~%~%


# Rework to have response
function fitda!(dr::DaResp, dp::RdaPred{RegDiscr})
	println("Regularized Discriminant Fit")
	ng = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centermatrix(dp.X,dp.means,dr.y.refs)
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
		dp.discr.coef[k,:] = dp.means[k,:] * dp.discr.whiten[:,:,k]		# mu_k in the sphered coordinates
		dp.discr.intercept[k] = -0.5*sum(dp.discr.coef[k,:] .^ 2) + dp.logpr[k]	# Constant term in quad discr function in sphered coordinates
	end
end

function predict(dp::Union(RdaPred{RegDiscr},RdaPred{QuadDiscr}), X::Matrix{Float64})
	n,p = size(X)
	ng = length(dp.logpr)
	Zk = Array(Float64,n,p)
	P = Array(Float64, n, ng)
	for k=1:ng
		Zk = X * dp.discr.whiten[:,:,k]
		P[:,k] = mapslices(sum, Zk .* (dp.discr.coef[k,:] .- 0.5*Zk), 2) .+ dp.discr.intercept[k]
	end
	println("This is P:")
	println(P)
	return mapslices(indmax,P,2)
end


function fitda!(dr::DaResp, dp::RdaPred{QuadDiscr})
	println("Quadratic Discriminant Analysis")
	ng = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centermatrix(dp.X,dp.means,dr.y.refs)
	for k = 1:ng
		class_k = find(dr.y.refs .==k)
		s, V = svd(Xc[class_k,:])[2:3]
		if dp.discr.gamma != 0 
			# Shrink towards (I * Average Eigenvalue)
			s = (s .^ 2)/(dr.counts[k]-1) .* (1-dp.discr.gamma) .+ (dp.discr.gamma * sum(s) / p)
		else	# No shrinkage
			s = (s .^ 2)/(dr.counts[k]-1)	# Check division properly
		end
		dp.discr.whiten[:,:,k] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
		dp.discr.coef[k,:] = dp.means[k,:] * dp.discr.whiten[:,:,k]			# mu_k in the sphered coordinates
		dp.discr.intercept[k] = -0.5*sum(dp.discr.coef[k,:] .^ 2) + dp.logpr[k]		# Constant term in quad discr function in sphered coordinates
	end
end

function fitda!(dr::DaResp, dp::RdaPred{LinDiscr}, rrlda=true)
	println("Linear Discriminant Analysis")
	ng = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centermatrix(dp.X,dp.means,dr.y.refs)
	#elseif isa(dp.discr, LinDiscr)
		# Do lda algorithm
		if rrlda == true
			# Do between variance calculation
		end
	#else
	#	error("Not a valid RdaPred subtype")
	#end
end




# %~%~%~%~%~%~%~%~%~% Wrapper Functions %~%~%~%~%~%~%~%~%


function rda(f::Formula, df::AbstractDataFrame; priors::Vector{Float64}=Float64[], lambda=0.5, gamma=0)
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
		dp = RdaPred{LinDiscr}(X, groupmeans(dr,X), LinDiscr(Array(Float64,p,p), gamma), log(pr))
	elseif lambda == 0
		dp = RdaPred{QuadDiscr}(X, groupmeans(dr,X), QuadDiscr(Array(Float64,p,p,k), gamma), log(pr))
	else
		discr = RegDiscr(Array(Float64,p,p,k), lambda, gamma, Array(Float64,k,p), Array(Float64,k))
		println("typeof(discr):")
		println(typeof(discr))
		dp = RdaPred{RegDiscr}(X, groupmeans(dr,X), discr, log(pr))
	end
	fitda!(dr,dp)
	return DaModel(mf,dr,dp,f)
end

function nclasses(mod::DaModel)
	return length(mod.dr.counts)
end

# Multiple Dispatch and alternatives
#function lda(f, df; priors=FP[], gamma=0) = rda(f, df; priors=priors, lambda=1, gamma=gamma)
#function qda(f, df; priors=FP[], gamma=0) = rda(f, df; priors=priors, lambda=0, gamma=gamma)


#function predict{T<:DisAnalysisModel}(mod::T, X::Matrix)
#	ng = length(mod.dr.counts)
#	n,p = size(X)
#	disf = Array(Float64,n,ng)
#	for k = 1:ng
#		disf[:,k] = mapslices(x-> norm((x .- mod.dp.means[k,:]) * mod.dp.whiten[:,:,k])^2, X, 2)
#	end
#	return disf
#end


