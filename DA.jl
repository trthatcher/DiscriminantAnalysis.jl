using DataFrames

module DA
	
	using DataFrames

	import DataFrames: ModelFrame, ModelMatrix, model_response, PooledDataVector, PooledDataArray

	export			# types
		DaResp,
		LdaMod,
		QdaMod,
		RdaMod,

				# functions
		rda,	# Regularized Discriminant Analysis (qda to lda shrinkage)
		lda,	# Linear Discriminant Analysis (with ridge analogue)
		qda,	# Quadratic Discriminant Analysis (with ridge analogue)
		predict,
		nclasses,

		groupmeans,
		centermatrix

	typealias FP FloatingPoint

	abstract ModResp

	abstract DaPred

	abstract DaModel

	include("rdafit.jl")
	include("kfdfit.jl")

end # module



