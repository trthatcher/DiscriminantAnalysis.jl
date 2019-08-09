module DiscriminantAnalysis
    using StatsBase, LinearAlgebra

    abstract type DiscriminantModel <: StatsBase.StatisticalModel end

    include("common.jl")
end