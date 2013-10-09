using DataFrames

data = readcsv("iris.csv")

predictor = data[:,1:4]

classes =  data[:,5] # map(x -> convert(Integer x), data[:,5])
classes = map(x -> convert(Integer, x), classes)


gamma = 0
lambda = 0.5

classlist = unique(classes)
k = length(classlist)
n,p = size(predictor)

sigma_k = Array(Float64, p, p, k)
mu_k = Array(Float64, p, k)
priors = Array(Float64, k)

sigma = Array(Float64, p, p)
sigma = cov(predictor)


for i in (1:k)
	class_indices = find(classes .== i)
	priors[i] = length(class_indices) / n
	mu_k[:,i] = vec(mean(predictor[class_indices,:], 1))
	sigma_k[:,:,i] = cov(predictor[class_indices,:])
end

sigma2_k = Array(Float64, p, p, k)

# Approximately correct for now
for i in (1:k)
	class_indices = find(classes .== i)
	mu_k[:,i] = vec(mean(predictor, 1))
	sigma2_k[:,:,i]	= (1-lambda) * sigma_k[:,:,i] + lambda * sigma
end


#############################

classlist = unique(classes)
k = length(classlist)
n,p = size(predictor)

mu_k = Array(Float64, p, k)
priors = Array(Float64, k)

sigma = Array(Float64, p, p)
sigma = cov(predictor)



if lambda == 1 # LDA
	mu_k[:,i] = vec(mean(predictor[class_indices,:], 1))

else
	sigma_k = Array(Float64, p, p, k)
	if lambda == 0 # QDA

	else # RDA
		for i in 1:k
			class_k = find(classes .== i)
			mu_k[:,i] = mean(predictor[class_k,:],1)
			for j in 1:n
				
			end
		end

end



