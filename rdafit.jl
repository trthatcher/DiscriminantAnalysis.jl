type DaResp{T<:FP} <: ModResp
	y::PooledDataArray	# Response vector
	priors::Vector{T}	# Prior weights
	counts::Vector{Int64}	# Prior observation counts
	function DaResp(y::PooledDataArray, priors::Vector{T})
		length(priors) == length(levels(y)) || error("Length mismatch priors/levels")
		n = length(y)
		c = zeros(Int64,n)
		c[y.refs[i]] += 1 for i = 1:n
		new(y, priors, c)
	end
end

type RDisAnalysis{T<:FP} <: DisAnalysis
	X::Matrix{T}
	means::Matrix{T}
	whiten::Array{T,3}
	logpr::Vector{T}
	lambda::T
	gamma::Real
	function RDisAnalysis(r::DaResp, X::Matrix{T}, lambda::Float64, gamma::Real=0)
		k = length(r.counts)
		n, p = size(X)
		means = groupmeans(r, X)
		whiten = Array(Float64,p,p,k)
		logpr = log(r.priors)
		new(X,means,whiten,logpr,lambda,gamma)
	end
end

type LDisAnalysis{T<:FP} <: DisAnalysis
	X::Matrix{T}
	means::Matrix{T}
	whiten::Matrix{T}
	logpr::Vector{T}
	gamma::Real
	function RDisAnalysis(r::DaResp, X::Matrix{T}, gamma::Real=0)
		k = length(r.counts)
		n, p = size(X)
		means = groupmeans(r, X)
		whiten = Array(Float64,p,p)
		logpr = log(r.priors)
		new(X,means,whiten,logpr,gamma)
	end
end

type QDisAnalysis{T<:FP} <: DisAnalysis
	X::Matrix{T}
	means::Matrix{T}
	whiten::Array{T,3}
	logpr::Vector{T}
	gamma::Real
	function RDisAnalysis(r::DaResp, X::Matrix{T}, gamma::Real=0)
		k = length(r.counts)
		n, p = size(X)
		means = groupmeans(r, X)
		whiten = Array(Float64,p,p,k)
		logpr = log(r.priors)
		new(X,means,whiten,logpr,gamma)
	end
end


type RdaMod <: DisAnalysisModel
	fr::ModelFrame
	dr::DaResp
	da::RDisAnalysis
	ff::Formula
end

type LdaMod <: DisAnalysisModel
	fr::ModelFrame
	dr::DaResp
	da::LDisAnalysis
	ff::Formula
end

type QdaMod <: DisAnalysisModel
	fr::ModelFrame
	dr::DaResp
	da::QDisAnalysis
	ff::Formula
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
	sd = sqrt(sd/(n-1))
	for i = 1:n	# BLAS improvement?
		Xc[i,:] = Xc[i,:] ./ sd
	end
	return (Xc, vec(sd))
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
function fitrda{T<:FP}(rr::DaResp,X::Matrix{T},lambda::T,gamma::Real)
	ng::Int64 = length(levels(rr.y))
	n::Int64, p::Int64 = size(X)
	nk::Vector{Int64}, mu_k = groupmeans(rr.y, X)
	Xc::Matrix{Float64}, sd::Vector{Float64} = centermatrix(X, mu_k, rr.y)
	Sigma = transpose(Xc) * Xc	# NOTE: Not weighted with n-1
	Sigma_k = Array(Float64,p,p)
	whiten_k = Array(Float64,p,p,ng)
	for k = 1:ng	# BLAS/LAPACK optimizations?
		class_k = find(rr.y.refs .== k)
		Sigma_k = transpose(Xc[class_k,:]) * Xc[class_k,:]
		Sigma_k = (1-lambda) * Sigma_k + lambda * Sigma		# Shrink towards Pooled covariance
		s, V = svd(Sigma_k)[2:3]
		s = s ./ ((1-lambda)*(nk[k]-1) + lambda*(n-ng))
		if gamma != 0
			s = s .* (1-gamma) .+ (gamma * trace(Sigma_k) / p)	# Shrink towards (I * Average Eigenvalue)
		end
		whiten_k[:,:,k] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
	end
	return (mu_k,whiten_k)
end

# Rework to have response
function fitda!{T<:FP}(dr::DaResp, da::RDisAnalysis)
	ng = length(levels(dr.counts))
	n, p = size(da.X)
	Xc, sd = centermatrix(da.X,da.means,dr.y.refs)
	Sigma = transpose(Xc) * Xc	# NOTE: Not weighted with n-1
	Sigma_k = Array(Float64,p,p)
	for k = 1:ng	# BLAS/LAPACK optimizations?
		class_k = find(dr.y.refs .== k)
		Sigma_k = transpose(Xc[class_k,:]) * Xc[class_k,:]
		Sigma_k = (1-lambda) * Sigma_k + lambda * Sigma		# Shrink towards Pooled covariance
		s, V = svd(Sigma_k)[2:3]
		s = s ./ ((1-lambda)*(nk[k]-1) + lambda*(n-ng))
		if gamma != 0
			s = s .* (1-gamma) .+ (gamma * trace(Sigma_k) / p)	# Shrink towards (I * Average Eigenvalue)
		end
		da.whiten[:,:,k] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
	end
end




# %~%~%~%~%~%~%~%~%~% Wrapper Functions %~%~%~%~%~%~%~%~%


function rda(f::Formula, df::AbstractDataFrame; priors::Vector{Float64}=Float64[], lambda::Real=0.5, gamma::Real=0)
	mf::ModelFrame = ModelFrame(f, df)
	mm::Matrix{Float64} = ModelMatrix(mf).m[:,2:]
	y::PooledDataVector = PooledDataArray(model_response(mf)) 	# NOTE: pooled conversion done INSIDE rda in case df leaves out factor levels
	n = length(y); k = length(levels(y)); lp = length(priors)
	lp == 0 || lp == k || error("length(priors) = $lp should be 0 or $k")
	p = lp == k ? copy(priors) : ones(Float64, k)/k
	rr = DaResp{Float64}(y, p)
	if lambda == 1
		da = fitlda()
		res = LdaMod(mf,rr,da,f,lambda,gamma)
	elseif lambda == 0
		da = fitqda()
		res = QdaMod(mf,rr,da,f,lambda,gamma)
	else
		a,b = fitrda(rr,mm, lambda, gamma)
		da = RDisAnalysis(a, b, log(priors))
		res = (rr, da) #RdaMod(mf,rr,da,f,lambda,gamma)
	end
end

# Multiple Dispatch and alternatives
#function lda(f, df; priors=FP[], gamma=0) = rda(f, df; priors=priors, lambda=1, gamma=gamma)
#function qda(f, df; priors=FP[], gamma=0) = rda(f, df; priors=priors, lambda=0, gamma=gamma)



