n_k = [6; 8; 10]
k = length(n_k)
p = 3
y = vcat([Int64[i for j = 1:n_k[i]] for i = 1:k]...)
X = vcat([rand(n_k[i], p) .+ (10rand(1,p) .- 5) for i = 1:k]...)
σ = sortperm(rand(sum(n_k)))
y = y[σ]
X = X[σ,:]
M = MOD.class_means(X, y)
Xc = MOD.center_classes!(copy(X), M, y)
w_σ = 1.0 ./ vec(sqrt(var(Xc, 1)))
H = MOD.scale!(copy(Xc), w_σ)
H_k = [H[y .== i,:] for i = 1:k]
Σ_k = [H_k[i]'H_k[i]/(n_k[i]-1) for i = 1:k]

info("Testing ", MOD.class_covariances)
for T in FloatingPointTypes
    for U in IntegerTypes
        H_tmp = convert(Array{T}, H)
        y_tmp = convert(Array{U}, y)
        Σ_k_tmp = MOD.class_covariances(H_tmp, y_tmp)
        for i = 1:k
            @test_approx_eq Σ_k_tmp[i] convert(Array{T}, Σ_k[i])
        end
    end
end

info("Testing ", MOD.class_whiteners!)
for T in FloatingPointTypes
    for λ in (zero(T), convert(T, 0.5), one(T))
        Σ_k_tmp = convert(Vector{Matrix{T}}, Σ_k)
        W_k_tmp = MOD.class_whiteners!(deepcopy(Σ_k_tmp), λ)
        for i = 1:k
            S = (1-λ)*Σ_k_tmp[i] + λ*(trace(Σ_k_tmp[i])/p)*I
            W = W_k_tmp[i]
            @test_approx_eq W'S*W diagm(ones(T,p)) 
        end
    end
end
