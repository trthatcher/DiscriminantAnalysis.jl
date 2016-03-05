n_k = [6; 8; 10]
k = length(n_k)
n = sum(n_k)
p = 3

refs = vcat([Int64[i for j = 1:n_k[i]] for i = 1:k]...)
Z = vcat([2*rand(n_k[i], p) .+ 3*(i-2) for i = 1:k]...)  # Linearly separable
σ = sortperm(rand(sum(n_k)))

y = MOD.RefVector(refs[σ])
X = Z[σ,:]
M = MOD.class_means(X, y)
H = MOD.center_classes!(copy(X), M, y)
Σ = H'H/(n-1)
priors = ones(k)./k

info("Testing ", MOD.lda!)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    M_tmp = convert(Matrix{T}, M)
    H_tmp = convert(Matrix{T}, H)
    Σ_tmp = convert(Matrix{T}, Σ)

    W_tmp = MOD.lda!(copy(X_tmp), M_tmp, y, Nullable{T}())
    @test_approx_eq W_tmp'*Σ*W_tmp eye(T,p)

    for γ in (zero(T), convert(T, 0.25), convert(T, 0.75), one(T))
        W_tmp = MOD.lda!(copy(X_tmp), M_tmp, y, Nullable(γ))
        S = (1-γ)*Σ_tmp + (γ/p)*trace(Σ_tmp)*I  # gamma-regularization

        @test_approx_eq W_tmp'*S*W_tmp eye(T,p)
    end
end

info("Testing ", MOD.cda!)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    M_tmp = convert(Matrix{T}, M)
    H_tmp = convert(Matrix{T}, H)
    Σ_tmp = convert(Matrix{T}, Σ)
    priors_tmp = convert(Vector{T}, priors)

    W_tmp = MOD.cda!(copy(X_tmp), copy(M_tmp), y, Nullable{T}(), priors_tmp)
    @test_approx_eq W_tmp'*Σ*W_tmp eye(T,k-1)

    for γ in (zero(T), convert(T, 0.25), convert(T, 0.75), one(T))
        W_tmp = MOD.cda!(copy(X_tmp), copy(M_tmp), y, Nullable(γ), priors_tmp)
        S = (1-γ)*Σ_tmp + (γ/p)*trace(Σ_tmp)*I  # gamma-regularization

        @test_approx_eq W_tmp'*S*W_tmp eye(T,k-1)
    end
end




#=
info("Testing ", MOD.lda)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    M_tmp = convert(Matrix{T}, M)
    for U in IntegerTypes
        y_tmp = convert(Vector{U}, y)
        for γ in (zero(T), convert(T, 0.5), one(T))
            W_tmp = MOD.lda!(copy(X_tmp), M_tmp, y_tmp, γ)
            Model = MOD.lda(copy(X_tmp), y_tmp, gamma = γ)
            @test_approx_eq W_tmp Model.W
        end
    end
end


info("Testing ", MOD.classify_lda)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    M_tmp = convert(Matrix{T}, M)
    priors = convert(Vector{T}, [1/k for i = 1:k])
    for U in IntegerTypes
        y_tmp = convert(Vector{U}, y)
        Model1 = MOD.lda(X_tmp, y_tmp, gamma = zero(T))
        Model2 = MOD.cda(X_tmp, y_tmp, gamma = zero(T))
        @test all(y_tmp .== MOD.classify_lda(Model1.W, Model1.M, priors, X_tmp))
        @test all(y_tmp .== MOD.classify_lda(Model2.W, Model2.M, priors, X_tmp))
        @test all(y_tmp .== classify(Model1, X_tmp))
        @test all(y_tmp .== classify(Model2, X_tmp))
    end
end
=#
