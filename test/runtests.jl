using Test, LinearAlgebra, Statistics, DiscriminantAnalysis

const DA = DiscriminantAnalysis

@testset "Common" begin
    include("test_common.jl")
end

#MOD = DiscriminantAnalysis
#
#FloatingPointTypes = (Float32, Float64)
#IntegerTypes = (Int32, Int64)
#
#function sampledata{U<:Integer}(n_k::Vector{U}, p::Integer)
#    p >= 3 || error("need at least 3 dimensions")
#    all(n_k .> p) || error("Need p+1 obs per group")
#    k = length(n_k)
#    Z = vcat([vcat(vcat(eye(Float64,p), Float64[ones(p-1); 0]')/2, 
#                   rand(Float64, n_k[i]-(p+1), p)) .+ i for i = 1:k]...)
#    refs = vcat([Int64[i for j = 1:n_k[i]] for i = 1:k]...)
#    σ = sortperm(rand(sum(n_k)))
#    (MOD.RefVector(refs[σ]), Z[σ,:])
#end

#include("test_lda.jl")
#include("test_qda.jl")
