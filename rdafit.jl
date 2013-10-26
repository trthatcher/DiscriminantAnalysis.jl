type DaResp{T<:FP} <: ModResp
	y::PooledDataArray	# Response vector
	priors::Vector{T}	# Prior weights
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

type RegDiscr{T<:FP} <: Discr
	whiten::Array{T,3}
	lambda::T
	gamma::Real
end

type LinDiscr{T<:FP} <: Discr
	whiten::Matrix{T}
	gamma::Real
end

type QuadDiscr{T<:FP} <: Discr
	whiten::Array{T,3}
	gamma::Real
end

type RdaPred{T<:FP} <: DaPred
	X::Matrix{T}
	means::Matrix{T}
	discr::Discr
	logpr::Vector{T}
	function RdaPred(r::DaResp, X::Matrix{T}, lambda, gamma=0)
		k = length(r.counts)
		n, p = size(X)
		means = groupmeans(r, X)
		logpr = log(r.priors)
		if lambda == 1
			discr = LinDiscr(Array(Float64,p,p), gamma)
		elseif lambda == 0
			discr = QuadDiscr(Array(Float64,p,p,k), gamma)
		else
			discr = RegDiscr(Array(Float64,p,p,k), lambda, gamma)
		end
		new(X,means,discr,logpr)
	end

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
function fitda!(dr::DaResp, dp::RdaPred)
	ng = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centermatrix(dp.X,dp.means,dr.y.refs)
	Sigma = transpose(Xc) * Xc	# NOTE: Not weighted with n-1
	Sigma_k = Array(Float64,p,p)
	if isa(dp.discr, RegDiscr) 
		for k = 1:ng	# BLAS/LAPACK optimizations?
			class_k = find(dr.y.refs .== k)
			Sigma_k = transpose(Xc[class_k,:]) * Xc[class_k,:]
			Sigma_k = (1-dp.discr.lambda) * Sigma_k + dp.discr.lambda * Sigma		# Shrink towards Pooled covariance		
			s::Vector{Float64}, V::Matrix{Float64} = svd(Sigma_k)[2:3]
			s = s ./ ((1-dp.discr.lambda)*(dr.counts[k]-1) + dp.discr.lambda*(n-ng))
			if dp.discr.gamma != 0
				s = s .* (1-dp.discr.gamma) .+ (dp.discr.gamma * trace(Sigma_k) / p)	# Shrink towards (I * Average Eigenvalue)
			end
			dp.discr.whiten[:,:,k] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
		end
	elseif isa(dp.discr, LinDiscr)
		
	else
		error("Not a valid RdaPred subtype")
	end
end




# %~%~%~%~%~%~%~%~%~% Wrapper Functions %~%~%~%~%~%~%~%~%


function rda(f::Formula, df::AbstractDataFrame; priors::Vector{Float64}=Float64[], lambda=0.5, gamma=0)
	mf::ModelFrame = ModelFrame(f, df)
	mm::Matrix{Float64} = ModelMatrix(mf).m[:,2:]
	y::PooledDataVector = PooledDataArray(model_response(mf)) 	# NOTE: pooled conversion done INSIDE rda in case df leaves out factor levels
	n = length(y)
	k = length(levels(y))
	lp = length(priors)
		lp == 0 || lp == k || error("length(priors) = $lp should be 0 or $k")
	pr = lp == k ? copy(priors) : ones(Float64, k)/k
	dr = DaResp{Float64}(y, pr)
	dp = RdaPred{Float64}(dr, mm, lambda, gamma)
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


