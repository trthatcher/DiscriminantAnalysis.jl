using DataFrames, DataArrays

module DiscriminantAnalysis
	
	using DataFrames, DataArrays

	import DataFrames: ModelFrame, ModelMatrix, model_response
	import DataArrays: PooledDataArray, levels, index_to_level

	export	######### TYPES
		DaModel,
		DaResp,
		DaPred,
		RdaPred,

		######### FUNCTIONS
		rda,	# Regularized Discriminant Analysis (qda to lda shrinkage)
		lda,	# Linear Discriminant Analysis (with ridge analogue)
		qda,	# Quadratic Discriminant Analysis (with ridge analogue)

		predict,  # Predict using a dataframe or matrix and a discriminant model, returns a pooled data vector
		classes,  # Return the class names in the model
		priors,   # List the prior probabilities
		counts,   # Return the frequency counts for each class
		formula,  # Return the formula that defines the model
		scaling,  # Same as whiten
		whiten,   # The transformation matrix that spheres the data
		means,    # The class means matrix, each row corresponds to a class mean
		logpriors, # Logarithm of the prior probabilities
		gamma,    # The gamma regularization parameter value (similar to ridge regression)
		lambda    # The lambda regularization parameter (shrinks the class covariance matrix to the pooled covariance matrix)

	include("types.jl")
	include("rdafit.jl")

end # module



