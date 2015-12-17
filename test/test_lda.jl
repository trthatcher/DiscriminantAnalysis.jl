n_k = [6; 8; 10]
k = length(n_k)
n = sum(n_k)
p = 3
y = vcat([Int64[i for j = 1:n_k[i]] for i = 1:k]...)
X = vcat([2*rand(n_k[i], p) .+ 3*(i-2) for i = 1:k]...)  # Linearly separable
# Permute Data
    σ = sortperm(rand(sum(n_k)))
    y = y[σ]
    X = X[σ,:]
M = MOD.class_means(X, y)
Xc = MOD.center_classes!(copy(X), M, y)
w_σ = 1.0 ./ vec(sqrt(var(Xc, 1)))
H = MOD.scale!(copy(Xc), w_σ)
Σ = H'H/(n-1)

info("Testing ", MOD.lda!)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    M_tmp = convert(Matrix{T}, M)
    Xc_tmp = convert(Matrix{T}, Xc)
    H_tmp = convert(Matrix{T}, H)
    Σ_tmp = convert(Matrix{T}, Σ)
    σ = convert(Vector{T}, 1 ./ w_σ)
    for U in IntegerTypes
        y_tmp = convert(Vector{U}, y)
        for γ in (zero(T), convert(T, 0.5), one(T))
            W_tmp = MOD.lda!(copy(X_tmp), M_tmp, y, γ)
            S = (1-γ)*Σ_tmp + (γ/p)*trace(Σ_tmp)*I  # gamma-regularization
            W = σ .* W_tmp                          # Remove scaling transform
            @test_approx_eq W'S*W diagm(ones(T,p))
            if γ == 0
                U = H_tmp * W
                @test_approx_eq U'U/(n-1) diagm(ones(T,p))
            end
        end
    end
end
