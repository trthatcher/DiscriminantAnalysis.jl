using DataFrames

data = readcsv("iris.csv")

x = data[:,1:4]

y = map(x -> convert(Integer, x), data[:,5])

w = ones(150)


gamma = 0
lambda = 0.8

#############################

# Input:
#   x: n by p matrix of predictors (one column per predictor)
#   y: n by 1 response vector of classes
#   w: n by 1 vector of observation weights (default weight is 1)
#   gamma: regularization parameter (shrink towards identity matrix)
#   lambda: regularization parameter (shrink QDA towards LDA)
#   priors: class prior weights

classlist = unique(y)
k = length(classlist)
n,p = size(x_mat)

mu_k = Array(Float64, p, k)
priors = Array(Float64, k)

if lambda == 1 # LDA
	println("Branch: LDA")
	sigma = Array(Float64, p, p)
	sigma = cov(x)
	for i in 1:k
		class_k = find(y .== i)
		mu_k[:,i] = vec(mean(x[class_k,:], 1))
	end
else
	sigma_k = Array(Float64, p, p, k)
	if lambda == 0 # QDA
		println("Branch: QDA")
		#class_indices = find(classes .== i)
		#mu_k[:,i] = vec(mean(predictor[class_indices,:], 1))
		#sigma_k[:,:,i] = cov(predictor[class_indices,:])
	else # RDA
		println("Branch: RDA")
		mu = Array(Float64,p)
		mu = vec(mean(x,1))		# Pooled mean vector for covariance estimation
		sigma = Array(Float64,p,p)
		sigma = cov(x)			# Pooled covariance matrix - is the coefficient correct?
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
end



