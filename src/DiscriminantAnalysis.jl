module DiscriminantAnalysis

    import Base: 
        LinAlg.BlasReal, 
        show,
        AbstractArray,
        size,
        linearindexing,
        getindex, 
        length, 
        convert

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
