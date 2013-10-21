#type DaResp{T<:FP} <: ModResp
#	y #::PooledDataArray	# Response vector
#	priors::Vector{T}	# Prior weights
#	function DaResp(y, priors::Vector{T})
#		length(priors) == length(levels(y)) || error("Length mismatch")
#		new(y, priors)
#	end
#end

type DaResp <: ModResp
	y	# Response vector
	priors::Vector{FP}	# Prior weights
	function DaResp(y, priors::Vector{FP})
		length(priors) == length(levels(y)) || error("Length mismatch")
		new(y, priors)
	end
end



type RDisAnalysis{T<:FP} <: DisAnalysis
	means::Matrix{T}
	whiten::Array{T,3}
	logpr::Vector{T}
end

type LDisAnalysis{T<:FP} <: DisAnalysis
	means::Matrix{T}
	whiten::Matrix{T}
	logpr::Vector{T}
end

type RdaMod <: DisAnalysisModel
	fr::ModelFrame
	rr::DaResp
	da::RDisAnalysis
	ff::Formula
	gg::Real
	ll::Real
end

type LdaMod <: DisAnalysisModel
	fr::ModelFrame
	rr::DaResp
	da::LDisAnalysis
	ff::Formula
	gg::Real
end

type QdaMod <: DisAnalysisModel
	fr::ModelFrame
	rr::DaResp
	da::RDisAnalysis
	ff::Formula
	gg::Real
end





# %~%~%~%~%~%~%~%~%~% Helper Functions %~%~%~%~%~%~%~%~%

# Find group means
# Don't pass NAs in the PDA or index 0 will be accessed
function groupmeans{T<:FP}(y::PooledDataVector, x::Matrix{T})
	n,p = size(x); k = length(levels(y))
	length(y) == n || error("Array lengths do not conform")
	g = zeros(FP, k, p); nk = zeros(Int64, k)
	for i = 1:n
		g[y.refs[i],:] += x[y.refs[i], :]
		nk[y.refs[i]] += 1
	end
	for i = 1:k
		g[i,:] = g[i,:] / nk[i]
	end
	return (nk, g)
end

# Center matrix by group mean
function centermatrix{T<:FP}(X::Matrix{T}, g::Matrix{T},y::PooledDataVector)
	n,p = size(X)
	Xc = Array(FP,n,p)
	sd = zeros(FP,1,p)
	for i = 1:n
		Xc[i,:] = X[i,:] - g[y.refs[i],:]
		sd += Xc[i,:].^2
	end
	sd = sqrt(sd / (n-1))
	for i = 1:n	# BLAS improvement?
		Xc[i,:] = Xc[i,:] ./ sd
	end
	return (Xc, sd)
end

function centermatrix{T<:FP}(X::Matrix{T}, g::Matrix{T})
	n,p = size(X)
	Xc = Array(FP,n,p)
	sd = zeros(FP,1,p)
	for i = 1:n
		Xc[i,:] -= g[1,:]
		sd += Xc[i,:].^2
	end
	sd = sqrt(sd / (n-1))
	for i = 1:n	# BLAS improvement?
		Xc[i,:] = Xc[i,:] ./ sd
	end
	return (Xc, sd)
end


# %~%~%~%~%~%~%~%~%~% Fitting Functions %~%~%~%~%~%~%~%~%


# Fit LDA
function fitlda()
	true
end

# Fit QDA
function fitqda()
	true
end

# Fit RDA 
function fitrda{T<:FP}(rr::DaResp,mm::Matrix{T},lambda::T,gamma::Real)
	ng::Int64 = length(levels(rr.y))
	n::Int64,p::Int64 = size(mm)
	nk::Vector{Int64}, mu_k = groupmeans(rr.y, mm)
	Xc, sd = centermatrix(rr.y, mm, mu_k)
	Sigma = transpose(Xc) * Xc	# NOTE: Not weighted with n-1
	Sigma_k = Array{FP,p,p}
	whiten_k = Array{FP,p,p,lk}
	for k = 1:ng	# BLAS/LAPACK optimizations?
		class_k = find(rr::y == k)
		Sigma_k = transpose(Xc[class_k,:]) * Xc_k[class_k,:]
		Sigma_k = (1-lambda) * Sigma_k + lambda * Sigma		# Shrink towards Pooled covariance
			# Sigma_k = Sigma_k / ((1-lambda)*(nk[k]-1) + lambda*(n-p))
		s, V = svd(Sigma_k)[2:3]
		s = s ./ ((1-lambda)*(nk[k]-1) + lambda*(n-ng))
		if gamma != 0
			s = s .* (1-gamma) .+ (gamma * trace(Sigma_k) / p)	# Shrink towards (I * Average Eigenvalue)
		end
		whiten_k[:,:,k] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
	end
	return (mu_k,whiten_k)
end



# %~%~%~%~%~%~%~%~%~% Wrapper Functions %~%~%~%~%~%~%~%~%


function rda{T<:FP}(f::Formula, df::AbstractDataFrame; priors::Vector{T}=FP[], lambda::Real=0.5, gamma::Real=0)
	mf::ModelFrame = ModelFrame(f, df)
	mm::Matrix{T} = ModelMatrix(mf).m[:,2:]
	y::PooledDataVector = PooledDataArray(model_response(mf)) 	# NOTE: pooled conversion done INSIDE rda in case df leaves out factor levels
	n = length(y); k = length(levels(y)); lp = length(priors)
	lp == 0 || lp == k || error("length(priors) = $lp should be 0 or $k")
	p = lp == k ? copy(priors) : ones(FP, k)/k
	rr = DaResp(y, p)
	if lambda == 1
		da = fitlda()
		return LdaMod(mf,rr,da,f,lambda,gamma)
	elseif lambda == 0
		da = fitqda()
		return QdaMod(mf,rr,da,f,lambda,gamma)
	else
		da = RDisAnalysis{FP}(fitrda(rr,mm, lambda, gamma), log(priors))
		return rr #RdaMod(mf,rr,da,f,lambda,gamma)
	end
end

# Multiple Dispatch and alternatives
#function lda(f, df; priors=FP[], gamma=0) = rda(f, df; priors=priors, lambda=1, gamma=gamma)
#function qda(f, df; priors=FP[], gamma=0) = rda(f, df; priors=priors, lambda=0, gamma=gamma)



