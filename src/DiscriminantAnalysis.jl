module DiscriminantAnalysis
    using StatsBase, LinearAlgebra

    # Common functions across classifiers
    include("common.jl")
    include("whiten.jl")

    # Classifier-specific code
    """
        DiscriminantModel

    Abstract type representing a discriminant model
    """
    abstract type DiscriminantModel{T<:AbstractFloat} <: StatsBase.StatisticalModel end

    include("discriminant_parameters.jl")
    for classifier in ["linear"] #, "quadratic"]
        include("classifiers/$(classifier).jl")
    end
    include("discriminant_functions.jl")
end