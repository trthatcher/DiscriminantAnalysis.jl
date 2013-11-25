using DataFrames

module DA
	
	using DataFrames

	import DataFrames: ModelFrame, ModelMatrix, model_response
	import DataArrays: PooledDataArray, levels, index_to_level

	export	# TYPES
		DaModel,
		DaResp,
		DaPred,
		RdaPred,

		######### FUNCTIONS
		rda,	# Regularized Discriminant Analysis (qda to lda shrinkage)
		lda,	# Linear Discriminant Analysis (with ridge analogue)
		qda,	# Quadratic Discriminant Analysis (with ridge analogue)
		predict,
		nclasses,

		groupmeans,
		centermatrix,
		fitda!

	include("types.jl")
	include("rdafit.jl")
	include("kfdfit.jl")

end # module



