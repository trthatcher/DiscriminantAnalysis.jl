using Test, LinearAlgebra, Statistics, DiscriminantAnalysis

const DA = DiscriminantAnalysis

# Sample normal data with mean zero and identity covariance
const Z = Float64[-0.2884140789858677    0.49399608183734783;
                   1.0936474817278636   -1.4507981791103175; 
                   1.2969519076037297    0.9479307530188256;
                  -0.16154175807885898  -0.401940103471996;
                  -0.20316231241257554  -0.0422591725837902;
                  -2.652877626067871    -1.167321893012155;
                  -1.0914320855828834    1.0280818796238944;
                  -0.7390611230924757   -0.858068799662248;
                  -0.6501244750345416    0.9428868728570992;
                   1.968552290249652    -0.892564366428744;
                   1.3257097650749985   -0.616395132118126;
                   0.19838108524304318   0.5303751626248464;
                   0.16416056348880076  -1.0291861836447858;
                  -0.2843443066759517    1.555074022653521;
                   0.839624717314438    -0.029362232323235926;
                  -0.6584211456536395   -0.9742092387159742;
                   0.42000156961086865  -1.5438953845438064;
                   1.0347431607313056    0.15264626385110022;
                   0.057873436358933765  0.6216484547095644;
                   0.2388690480037279    1.9422059093023596;
                  -0.42922703496294135  -0.3002264078515958;
                   0.37777109846081014   1.8431573629118982;
                  -0.17693678202744545   0.1023133591460945;
                  -1.782944658967597    -0.0872658571887212;
                   0.1022012636744785   -0.7668231718810562]


function generate_data(T::Type{<:AbstractFloat}, n::Int, p::Int)
    ix = sortperm(rand(2n))

    y = repeat(1:2, inner = n)[ix]

    X1 = rand(T, n, p)
    X1 .-= mean(X1, dims = 1)
    X1 .+= 1
    X2 = rand(T, n, p)
    X2 .-= mean(X2, dims = 1)
    X2 .-= 1
    
    X = [X1; X2][ix, :]

    M = [mean(X1, dims = 1); mean(X2, dims = 1)]

    (X, y, M)
end


@testset "common.jl" begin
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
