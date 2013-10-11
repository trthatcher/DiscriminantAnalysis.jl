using DataFrames

module DA
	
	using DataFrames

	import DataFrames: ModelFrame, ModelMatrix, model_response

	export			# types
		RdaResp,
		RdaMod,

				# functions
		rda,	# Regularized Discriminant Analysis (qda to lda shrinkage)
		lda,	# Linear Discriminant Analysis
		rlda,	# Regularized Linear Discriminant Analysis (ridge analogue)
		qda,	# Quadratic Discriminant Analysis
		rqda	# Regularized Quadratic Discriminant Analysis (ridge analogue)

	abstract ModResp

	abstract DisAnalysis

	include("rda.jl")
	include("kfd.jl")

end # module



