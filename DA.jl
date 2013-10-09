using DataFrames

data = readcsv("iris.csv")

x_mat = data[:,1:4]

classes =  data[:,5] # map(x -> convert(Integer x), data[:,5])
classes = map(x -> convert(Integer, x), classes)


gamma = 0
lambda = 0.5

#############################

classlist = unique(classes)
k = length(classlist)
n,p = size(x_mat)

mu_k = Array(Float64, p, k)
priors = Array(Float64, k)

sigma = Array(Float64, p, p)
sigma = cov(x_mat)



if lambda == 1 # LDA
	#mu_k[:,i] = vec(mean(x_mat[class_k,:], 1))

else
	sigma_k = Array(Float64, p, p, k)
	if lambda == 0 # QDA
		#class_indices = find(classes .== i)
		#priors[i] = length(class_indices) / n
		#mu_k[:,i] = vec(mean(predictor[class_indices,:], 1))
		#sigma_k[:,:,i] = cov(predictor[class_indices,:])
	else # RDA
		mu = Array(Float64,p)
		mu = vec(mean(x_mat,1))
		sigma = Array(Float64,p,p)
		sigma = cov(x_mat)
		nk = Array(Int64, k)
		for i in 1:k
			class_k = find(classes .== i)
			nk[i] = length(class_k)
			mu_k[:,i] = vec(mean(x_mat[class_k,:],1))
			xc_mat = Array(Float64,nk[i],p)
			for j in 1:nk
				xc_mat[j,:] = x_mat[class_k[j],:] - mu_k[:,i]'	
			end
			sigma_k[:,:,i] = ((1-lambda)*(xc_mat' * xc_mat) + lambda * sigma)/((1-lambda)*nk + lambda*n)
		end
	end
end



