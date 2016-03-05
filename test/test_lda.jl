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

info("Testing ", MOD.lda)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    M_tmp = convert(Matrix{T}, M)

    model = MOD.lda(X_tmp, y, gamma=Nullable{T}())
    W_tmp = MOD.lda!(copy(X_tmp), copy(M_tmp), y, Nullable{T}())
    @test model.is_cda == false
    @test_approx_eq model.W W_tmp
    @test_approx_eq model.M M_tmp

    model = MOD.lda(X_tmp, refs, gamma=Nullable{T}())  # Check reference passing
    W_tmp = MOD.lda!(copy(X_tmp), copy(M_tmp), y, Nullable{T}())
    @test model.is_cda == false
    @test_approx_eq model.W W_tmp
    @test_approx_eq model.M M_tmp

    for γ in (zero(T), convert(T, 0.25), convert(T, 0.75), one(T))
        model = MOD.lda(X_tmp, y, gamma=Nullable(γ))
        W_tmp = MOD.lda!(copy(X_tmp), copy(M_tmp), y, Nullable{T}(γ))

        @test model.is_cda == false
        @test_approx_eq model.W W_tmp
        @test_approx_eq model.M M_tmp
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

info("Testing ", MOD.cda)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    M_tmp = convert(Matrix{T}, M)
    priors_tmp = convert(Vector{T}, priors)

    model = MOD.cda(X_tmp, y, gamma=Nullable{T}())
    W_tmp = MOD.cda!(copy(X_tmp), copy(M_tmp), y, Nullable{T}(), priors_tmp)
    @test model.is_cda == true
    @test_approx_eq model.W W_tmp
    @test_approx_eq model.M M_tmp

    model = MOD.cda(X_tmp, refs, gamma=Nullable{T}())  # Check reference passing
    W_tmp = MOD.cda!(copy(X_tmp), copy(M_tmp), y, Nullable{T}(), priors_tmp)
    @test model.is_cda == true
    @test_approx_eq model.W W_tmp
    @test_approx_eq model.M M_tmp

    for γ in (zero(T), convert(T, 0.25), convert(T, 0.75), one(T))
        model = MOD.cda(X_tmp, y, gamma=Nullable(γ))
        W_tmp = MOD.cda!(copy(X_tmp), copy(M_tmp), y, Nullable{T}(γ), priors_tmp)

        @test model.is_cda == true
        @test_approx_eq model.W W_tmp
        @test_approx_eq model.M M_tmp
    end
end

info("Testing ", MOD.discriminants_lda)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    priors_tmp = convert(Vector{T}, priors)

    model = lda(X_tmp, y)
    Z2 = hcat([MOD.dotrows((X_tmp .- M[i,:])*model.W) for i in eachindex(priors_tmp)]...)
    δ = -Z2/2 .+ log(priors_tmp)'

    @test_approx_eq δ MOD.discriminants(model, X_tmp)

    for γ in (zero(T), convert(T, 0.25), convert(T, 0.75), one(T))
        model = lda(X_tmp, y)
        Z2 = hcat([MOD.dotrows((X_tmp .- M[i,:])*model.W) for i in eachindex(priors_tmp)]...)
        δ = -Z2/2 .+ log(priors_tmp)'

        @test_approx_eq δ MOD.discriminants(model, X_tmp)
    end
end
