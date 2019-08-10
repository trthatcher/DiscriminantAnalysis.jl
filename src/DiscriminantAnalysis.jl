module DiscriminantAnalysis
    using StatsBase, LinearAlgebra

    abstract type DiscriminantModel{T<:AbstractFloat} <: StatsBase.StatisticalModel end

    include("common.jl")
    include("lda.jl")
end