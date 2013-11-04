using DataFrames

module DA
	
	using DataFrames

	import DataFrames: ModelFrame, ModelMatrix, model_response, PooledDataVector, PooledDataArray

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

	typealias FP FloatingPoint

	include("types.jl")
	include("rdafit.jl")
	include("kfdfit.jl")

end # module



