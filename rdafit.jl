type RdaResp
	y::Vector
	ylab::Vector
	wts::Vector	# prior weights
end

type RdaMod
	fr::ModelFrame
	rr::RdaResp
	dd::DisAnalysis	# Uses discriminant functions
	ff::Formula
end

function fitlda()
	classlist = unique(y)
	k = length(classlist)
	n,p = size(x_mat)

	mu_k = Array(Float64, p, k)

	sigma = Array(Float64, p, p)
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



function rda(f::Formula, df::AbstractDataFrame, lambda::Real, gamma::Real)
	mf = ModelFrame(f, df)
	# mm = 
	ModelMatrix(mf)
end

# Multiple Dispatch and alternatives
lda(f, df) = rda(f, df, 1, 0)
rlda(f, df, gamma) = rda(f, df, 1, gamma)
qda(f, df) = rda(f, df, 0, 0)
rqda(f, df, gamma) = rda(f, df, 0, gamma)



