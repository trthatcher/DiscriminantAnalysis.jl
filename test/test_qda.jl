n_k = [6; 8; 10]
k = length(n_k)
n = sum(n_k)
p = 3

refs = vcat([Int64[i for j = 1:n_k[i]] for i = 1:k]...)
Z = vcat([2*rand(n_k[i], p) .+ 3*(i-2) for i = 1:k]...)  # Linearly separable
σ = sortperm(rand(sum(n_k)))
refs = refs[σ]

y = MOD.RefVector(refs)
X = Z[σ,:]
M = MOD.class_means(X, y)
H = MOD.center_classes!(copy(X), M, y)
Σ = H'H/(n-1)
priors = ones(k)./k

info("Testing ", MOD.class_whiteners!)
for T in FloatingPointTypes
    H_tmp = convert(Array{T}, H)
    Σ_tmp = convert(Array{T}, Σ)
    f_k = one(T)./(MOD.class_counts(y) .- 1)
    Σ_k = Array{T,2}[MOD.gramian(H_tmp[y .== i,:], f_k[i], true) for i = 1:y.k]

    for λ in (zero(T), convert(T, 0.25), convert(T, 0.75), one(T))
        W_k = MOD.class_whiteners!(copy(H_tmp), y, Nullable{T}(), λ)
        for i in eachindex(Σ_k)
            S = (1-λ)*Σ_k[i] + λ*Σ_tmp
            @test_approx_eq eye(T,3) W_k[i]'*S*W_k[i]
        end

        for γ in (zero(T), convert(T, 0.25), convert(T, 0.75), one(T))
            W_k = MOD.class_whiteners!(copy(H_tmp), y, Nullable(γ), λ)
            for i in eachindex(Σ_k)
                S = (1-λ)*Σ_k[i] + λ*Σ_tmp
                S = (1-γ)*S + γ*trace(S)/p*I
                @test_approx_eq eye(T,3) W_k[i]'*S*W_k[i]
            end
        end
    end
end
