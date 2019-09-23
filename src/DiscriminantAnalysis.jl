module DiscriminantAnalysis
    using StatsBase, LinearAlgebra

    include("common.jl")
    include("whiten.jl")
    include("discriminant.jl")
    include("linear.jl")
    #include("quadratic.jl")
end