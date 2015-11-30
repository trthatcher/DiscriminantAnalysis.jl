module DiscriminantAnalysis

    using Base.LinAlg.BlasReal

    export
        lda,
        cda,
        qda,
        classify

    include("common.jl")
    include("lda.jl")
    include("qda.jl")

end # module
