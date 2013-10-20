type DaResp{T<:FP} <: ModResp
	y::PooledDataArray	# Response vector
	wts::Vector{T}		# Observation weights
	priors::Vector{T}	# Prior weights
	function DaResp(y::PooledDataArray, wts::Vector{T}, priors::Vector{T})
		lw = length(wts)
		lw == 0 || lw == length(y) || error("Length mismatch")
		length(priors) == length(levels(y)) || error("Length mismatch")
		new(y, wts, priors)
	end
end

type RDisAnalysis{T<:FP} <: DisAnalysis
	means::Matrix{T}
	whiten::Array{T,3}
	logpr::Vector{T}
end

type LDisAnalyis{T<:FP} <: DisAnalyis
	means::Matrix{T}
	whiten::Matrix{T}
	logpr::Vector{T}
end

type RdaMod <: DisAnalysisModel
	fr::ModelFrame
	rr::DaResp
	da::RDisAnalyis
	ff::Formula
	gg::Real
	ll::Real
end

type LdaMod <: DisAnalysisModel
	fr::ModelFrame
	rr::DaResp
	da::LDisAnalyis
	ff::Formula
	gg::Real
end

type QdaMod <: DisAnalysisModel
	fr::ModelFrame
	rr::DaResp
	da::RDisAnalyis
	ff::Formula
	gg::Real
end





# %~%~%~%~%~%~%~%~%~% Helper Functions %~%~%~%~%~%~%~%~%

# Find group means
# Don't pass NAs in the PDA or index 0 will be accessed
function groupmeans{T<:FP}(y::PooledDataArray, x::Matrix{T})
	n,p = size(x); k = length(levels(y))
	length(y) == n || error("Array lengths do not conform")
	g = zeros(FP, k, p); nk = zeros(Int64, k)
	for i = 1:n
		g[y[i],:] += x[y[i], :]
		nk[y[i]] += 1
	end
	for i = 1:k
		g[i,:] = g[i,:] / nk[i]
	end
	(nk, g)
end

# Center matrix by group mean
function centermatrix{T<:FP}(X::Matrix{T}, g::Matrix{T},y::PooledDataArray)
	n,p = size(X)
	Xc = Array(FP,n,p)
	sd = zeros(FP,1,p)
	for i = 1:n
		Xc[i,:] = X[i,:] - g[y[i],:]
		sd += Xc[i,:].^2
	end
	sd = sqrt(sd / (n-1))
	for i = 1:n	# BLAS improvement?
		Xc[i,:] = Xc[i,:] ./ sd
	(Xc, sd)
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
	(Xc, sd)
end


# %~%~%~%~%~%~%~%~%~% Fitting Functions %~%~%~%~%~%~%~%~%


# Fit LDA
function fitlda()
	true
end

# Fit QDA
functon fitqda()
	true
end

# Fit RDA 
function fitrda{T<:FP}(rr::DaResp,mm::Matrix{T},lambda::T,gamma::Real)
	ng::Int64 = length(levels(rr.y))
	n::Int64,p::Int64 = size(mm)
	nk::Vector{Int64}, mu_k = groupmeans(rr.y, mm)
	Xc, sd = centermatrix(rr.y, mm, mu_k)
	Sigma = transpose(Xc) * Xc	# NOTE: Not weighted with n-1
	Sigma_k = Array{FP,p,p)
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
	(mu_k,whiten_k)
end



# %~%~%~%~%~%~%~%~%~% Wrapper Functions %~%~%~%~%~%~%~%~%


function rda{T<:FP}(f::Formula, df::AbstractDataFrame; priors::Vector{T}=FP[], lambda::Real=0.5, gamma::Real=0)
	mf = ModelFrame(f, df)
	mm = ModelMatrix(mf)[:,2:]
	y = PooledDataArray(model_response(mf)) 	# NOTE: pooled conversion done INSIDE rda in case df leaves out factor levels
		n = length(y); k = length(levels(y)); lw = length(wts); lp = length(priors)
		lw == 0 || lw == n || error("length(wts) = $lw should be 0 or $n") 
	w = lw == 0 ? FP[] : copy(wts)/sum(wts)
		lp == 0 || lp == k || error("length(priors) = $lp should be 0 or $k")
	p = priors == k ? copy(priors) : ones(FP, k)/k
	rr = RdaResp(y, w, p)
	if lambda == 1
		da = fitlda()
	elseif lambda == 0
		da = fitqda()
	else
		da = RDisAnalysis(fitrda(rr,mm, lambda, gamma), log(priors))
		RdaMod(mf,rr,da,f,lambda,gamma)
	end
end

# Multiple Dispatch and alternatives
#lda(f, df) = rda(f, df, 1, 0)
#lda(f, df, gamma) = rda(f, df, 1, gamma)
#qda(f, df) = rda(f, df, 0, 0)
#qda(f, df, gamma) = rda(f, df, 0, gamma)



