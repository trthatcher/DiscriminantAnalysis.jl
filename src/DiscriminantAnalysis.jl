module DiscriminantAnalysis

    using Base.LinAlg.BlasReal

    export
        qda,
        predict

    include("common.jl")
    include("qda.jl")

end # module
