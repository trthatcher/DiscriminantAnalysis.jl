type DaResp{T<:FP} <: ModResp
	y::PooledDataArray	# Response vector
	wts::Vector{T}	# Observation weights
	priors::Vector	# Prior weights
end

type RdaMod <: DisAnalysisModel
	fr::ModelFrame
	rr::DaResp
	mm::Array	# Pooled mean
	mk::Array	# Class means
	ss::Array	# Pooled sigma
	sk::Array	# Class sigmas
	ff::Formula
end

type LdaMod <: DisAnalysisModel
	fr::ModelFrame
	rr::DaResp
	mm::Array	# Pooled mean
	mk::Array	# Class means
	ss::Array	# Pooled sigma
	ff::Formula
end



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


function fitqda()
	5
end

# Input:
#   x: n by p matrix of predictors (one column per predictor)
#   y: n by 1 response vector of classes
#   w: n by 1 vector of observation weights (default weight is 1)
#   gamma: regularization parameter (shrink towards identity matrix)
#   lambda: regularization parameter (shrink QDA towards LDA)
#   priors: class prior weights

function fitrda()
	classlist = unique(y)
	k = length(classlist)
	n,p = size(x_mat)

	mu_k = Array(Float64, p, k)

	mu = Array(Float64,p); mu = vec(mean(x,1))	# Pooled mean vector for covariance estimation
	sigma = Array(Float64,p,p); sigma = cov(x)	# Pooled covariance matrix - is the coefficient correct?
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
	mm = ModelMatrix(mf)
	y = PooledDataArray(model_response) 	# NOTE: pooled conversion done INSIDE rda in case df is spliced (and leaves out factor levels)
		n = length(y); k = length(levels(y)); lw = length(wts); lp = length(priors)
		lw == 0 || lw == n || error("length(wts) = $lw should be 0 or $n") 
	w = lw == 0 ? FP[] : copy(wts)/sum(wts)
		lp == 0 || lp == k || error("length(priors) = $lp should be 0 or $k")
	p = priors == k ? copy(priors) : ones(FP, k)/k
	rr = RdaResp(y, w, p)

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

