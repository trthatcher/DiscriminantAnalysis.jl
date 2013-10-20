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

type RDisAnalysis <: DisAnal
	means::Array
	whiten::Array
	logpr::Vector
end

type LDisAnalyis <: DisAnal
	means::Array
	whiten::Array
	logpr::Vector
end

type RdaMod <: DisAnalModel
	fr::ModelFrame
	rr::DaResp
	da::RDisAnalyis
	ff::Formula
end

type LdaMod <: DisAnalModel
	fr::ModelFrame
	rr::DaResp
	da::LDisAnalyis
	ff::Formula
end





# %~%~%~%~%~%~%~%~%~ Centering Functions %~%~%~%~%~%~%~%~%

# Find group means
# Don't pass NAs in the PDA or index 0 will be accessed
function groupmeans{T<:FP}(y::PooledDataArray, x::Array{T})
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
function centermatrix{T<:FP}(X::Array{T,2}, g::Array{T,2},y::PooledDataArray)
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
	Xc, sd
end

function centermatrix{T<:FP}(X::Array{T,2}, g::Array{T})
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
	Xc, sd
end



# Fit LDA with and without gamma

# Fit QDA with and without gamma (doesn't require covariance

function fitrda(rr::DaResp,mm::Array,lambda::FP,gamma::Real)
	ng = length(levels(rr.y))
	n,p = size(mm)
	nk, mu_k = groupmeans(rr.y, mm)
	Xc, sd = centermatrix(rr.y, mm, mu_k)
	Sigma = transpose(Xc) * Xc	# NOTE: Not weighted with n-1
	Sigma_k = Array{FP,p,p)
	whiten = Array{FP,p,p,lk}
	for k = 1:ng	# BLAS/LAPACK optimizations?
		class_k = find(rr::y == k)
		Sigma_k = transpose(Xc[class_k,:]) * Xc_k[class_k,:]
		Sigma_k = (1-lambda) * Sigma_k + lambda * Sigma		# Shrink towards Pooled covariance
			# Sigma_k = Sigma_k / ((1-lambda)*(nk[k]-1) + lambda*(n-p))
		s, V = svd(Sigma_k)[2:3]
		s = s ./ ((1-lambda)*(nk[k]-1) + lambda*(n-ng))
		if gamma != 0
			s = s .* (1-gamma) .+ (gamma * trace(Sigma_k) / p)	# Shrink towards I * Average Eigenvalue
		end
		whiten[:,:,k] = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
	end


	sigma = Array(FP,p,p)
	sigma = cov(x)	# Pooled covariance matrix - is the coefficient correct?
	nk = Array(Int64, k)		# Class counts
	for i in 1:k
		class_k = find(classes .== i)	# Indices for class k
		nk[i] = length(class_k)
		mu_k[:,i] = vec(mean(x[class_k,:],1))
		xc = Array(Float64,nk[i],p)
		for j in 1:nk[i]
			xc[j,:] = x[class_k[j],:] - mu_k[:,i]'	# Center the x matrix (mean has been computed)
		end
		sigma_k[:,:,i] = ((1-lambda)*(xc' * xc) + lambda * sigma)/((1-lambda)*nk[i] + lambda*n)
	end
end





function rda{T<:FP}(f::Formula, df::AbstractDataFrame; lambda::Real=0.5, gamma::Real=0, priors::Vector{T}=FP[], wts::Vector{T}=FP[])
	mf = ModelFrame(f, df)
	mm = ModelMatrix(mf)[:,2:]
	y = PooledDataArray(model_response(mf)) 	# NOTE: pooled conversion done INSIDE rda in case df is spliced (and leaves out factor levels)
		n = length(y); k = length(levels(y)); lw = length(wts); lp = length(priors)
		lw == 0 || lw == n || error("length(wts) = $lw should be 0 or $n") 
	w = lw == 0 ? FP[] : copy(wts)/sum(wts)
		lp == 0 || lp == k || error("length(priors) = $lp should be 0 or $k")
	p = priors == k ? copy(priors) : ones(FP, k)/k
	rr = RdaResp(y, w, p)
	if lambda == 1
		da = lda(
	else
		da = rda(
	end
end

# Multiple Dispatch and alternatives
lda(f, df) = rda(f, df, 1, 0)
lda(f, df, gamma) = rda(f, df, 1, gamma)
qda(f, df) = rda(f, df, 0, 0)
qda(f, df, gamma) = rda(f, df, 0, gamma)

#function rda{T <: FP}(f::Formula, df::AbstractDataFrame; lambda::Real = 0.0, gamma::Real = 0.0, priors::Union(Vector{T},Dict)=FP[], wts=FP[])
#	mf = ModelFrame(f, df)
#	mm = ModelMatrix(mf)
#		#mf.terms.response || error("Model formula one-sided")
#		#y = mf.df[:,1]
#		#isa(y, PooledDataArray) || error("Response vector is not a PooledDataArray")
#	y = PooledDataArray(model_response) # ISSUE: levels(iris[1:100,:]["Species"]) returns three levels when there's two
#	n = length(y); k = length(levels(y)); lw = length(wts)
#	if isa(priors, Dict)
#		# Check if the collection is complete
#		d = priors
#	else
#		lw = length(wts)
#		if lw == 0 
#			
#		#d = isa(priors, Vector{T}) ? Priors(levels(y), priors) : priors
#	end
#	rr = RdaResp(y, wts, d)
#end


# Create dictionary for priors
#function Priors{T<:FP}(class::Vector, weight::Vector{T})
#	n = length(class)
#	n == length(weight) || error("Class and prior mismatch")
#	{class[i] => weight[i] for i = 1:n}
#end



function fitlda()
	classlist = unique(y)
	k = length(classlist); n,p = size(x_mat)

	mu_k = Array(Float64, p, k); sigma = Array(Float64, p, p)
	sigma = cov(x)
	for i in 1:k
		class_k = find(y .== i)
		mu_k[:,i] = vec(mean(x[class_k,:], 1))
	end
	
end


