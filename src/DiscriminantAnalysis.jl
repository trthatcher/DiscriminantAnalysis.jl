module DiscriminantAnalysis

export
    class_covariances,
    class_whiteners!,
    qda!,
    qda,
    predict_qda,
    predict

include("common.jl")
include("qda.jl")

end # module
