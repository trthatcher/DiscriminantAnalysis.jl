n_k = [4; 7; 5];
k = length(n_k)
y = vcat([Int64[i for j = 1:n_k[i]] for i = 1:k]...);
p = 3;
X = vcat([rand(Float64, n_k[i], p) .+ (10rand(1,p) .- 5) for i = 1:k]...);

M = MOD.class_means(X, y)
H = MOD.center_rows!(copy(X), M, y)
w_σ = 1.0 ./ vec(sqrt(var(X, 1)))
MOD.scale!(H, w_σ)
H_k = [H[y .== i,:] for i = 1:k]

info("Testing ", MOD.class_covariances)
for T in FloatingPointTypes
    Σ_k = class_covariances(convert(Array{T}, copy(H)), y)
    for i = 1:k
        @test_approx_eq Σ_k[i] convert(Array{T}, H_k[i]'H_k[i]/(n_k[i]-1))
    end
end
