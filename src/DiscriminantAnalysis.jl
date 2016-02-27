module DiscriminantAnalysis

    using Base.LinAlg.BlasReal

    export
        lda,
        cda,
        qda,
        discriminants,
        classify

    include("common.jl")
    include("lda.jl")
    include("qda.jl")

end # module
