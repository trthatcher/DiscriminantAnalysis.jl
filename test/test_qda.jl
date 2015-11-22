n_k = [6; 8; 10]
k = length(n_k)
n = sum(n_k)
p = 3
y = vcat([Int64[i for j = 1:n_k[i]] for i = 1:k]...)
X = vcat([rand(n_k[i], p) .+ (10rand(1,p) .- 5) for i = 1:k]...)
σ = sortperm(rand(sum(n_k)))
y = y[σ]
X = X[σ,:]
M = MOD.class_means(X, y)
Xc = MOD.center_classes!(copy(X), M, y)
#w_σ = 1.0 ./ vec(sqrt(var(Xc, 1)))
#H = MOD.scale!(copy(Xc), w_σ)
H = Xc
Σ = H'H/(n-1)
H_k = [H[y .== i,:] for i = 1:k]
Σ_k = [H_k[i]'H_k[i]/(n_k[i]-1) for i = 1:k]

info("Testing ", MOD.class_covariances)
for T in FloatingPointTypes, U in IntegerTypes
    H_tmp = convert(Array{T}, H)
    y_tmp = convert(Array{U}, y)
    Σ_k_tmp = MOD.class_covariances(H_tmp, y_tmp)
    for i = 1:k
        @test_approx_eq Σ_k_tmp[i] convert(Array{T}, Σ_k[i])
    end
end

info("Testing ", MOD.class_whiteners!)
for T in FloatingPointTypes
    Σ_k_tmp = convert(Vector{Matrix{T}}, Σ_k)
    for γ in (zero(T), convert(T, 0.5), one(T))
        W_k_tmp = MOD.class_whiteners!(deepcopy(Σ_k_tmp), γ)
        for i = 1:k
            S = (1-γ)*Σ_k_tmp[i] + γ*(trace(Σ_k_tmp[i])/p)*I
            W = W_k_tmp[i]
            @test_approx_eq W'S*W diagm(ones(T,p)) 
        end
    end
end

info("Testing ", MOD.qda!)
for T in FloatingPointTypes, U in IntegerTypes
    X_tmp = copy(convert(Matrix{T}, X))
    y_tmp = convert(Vector{U}, y)
    M_tmp = convert(Matrix{T}, M)
    Xc_tmp = convert(Matrix{T}, Xc)
    Σ_tmp = convert(Matrix{T}, Σ)
    Σ_k_tmp = convert(Vector{Matrix{T}}, Σ_k)
    for λ in (zero(T), convert(T, 0.5), one(T)), γ in (zero(T), convert(T, 0.5), one(T))
        W_k_tmp = qda!(copy(X_tmp), M_tmp, y_tmp, λ, γ)
        for i = 1:k
            #S = Xc_tmp[y .== i,:]'Xc_tmp[y .== i,:]/(n_k[i]-1)
            #S = (1-λ)*S + λ*S
            S = (1-λ)*Σ_k_tmp[i] + λ*Σ_tmp  # lambda-regularization first
            S = (1-γ)*S + (γ/p)*trace(S)*I  # gamma-regularization second
            W = W_k_tmp[i]
            @test_approx_eq W'S*W diagm(ones(T,p))
            if λ == 0 && γ == 0
                U_k = Xc_tmp[y .== i,:] * W
                @test_approx_eq U_k'U_k/(n_k[i]-1) diagm(ones(T,p))
            end
        end
    end
end
