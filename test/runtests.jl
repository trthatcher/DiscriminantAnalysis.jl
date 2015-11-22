using DiscriminantAnalysis
using Base.Test

MOD = DiscriminantAnalysis

FloatingPointTypes = (Float32, Float64)
IntegerTypes = (Int32, Int64)

include("test_common.jl")
include("test_qda.jl")
