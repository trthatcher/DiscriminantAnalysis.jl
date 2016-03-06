n_k = [80; 100; 120]
k = length(n_k)
n = sum(n_k)
p = 3

y, X = sampledata(n_k, p)
M = MOD.class_means(X, y)
H = MOD.center_classes!(copy(X), M, y)
Σ = H'H/(n-1)
priors = ones(k)./k

info("Testing ", MOD.class_whiteners!)
for T in FloatingPointTypes
    H_tmp = convert(Array{T}, H)
    Σ_tmp = convert(Array{T}, Σ)
    f_k = one(T)./(MOD.class_counts(y) .- 1)
    Σ_k = Array{T,2}[MOD.gramian(H_tmp[y .== i,:], f_k[i]) for i = 1:y.k]

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
    for γ in (zero(T), convert(T, 0.25), convert(T, 0.75), one(T))
        W_k = MOD.class_whiteners!(copy(H_tmp), y, Nullable(γ))
        for i in eachindex(Σ_k)
            S = (1-γ)*Σ_k[i] + γ*trace(Σ_k[i])/p*I
            @test_approx_eq eye(T,3) W_k[i]'*S*W_k[i]
        end
    end
end

info("Testing ", MOD.qda!)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    M_tmp = convert(Matrix{T}, M)
    H_tmp = convert(Matrix{T}, H)

    γ = Nullable{T}()
    λ = Nullable{T}()
    W_k = MOD.qda!(copy(X_tmp), copy(M_tmp), y, γ, λ)
    W_k_tmp = MOD.class_whiteners!(copy(H_tmp), y, γ)
    for i in eachindex(W_k)
        @test_approx_eq W_k[i] W_k_tmp[i]
    end

    γ = Nullable(one(T)/3)
    λ = Nullable{T}()
    W_k = MOD.qda!(copy(X_tmp), copy(M_tmp), y, γ, λ)
    W_k_tmp = MOD.class_whiteners!(copy(H_tmp), y, γ)
    for i in eachindex(W_k)
        @test_approx_eq W_k[i] W_k_tmp[i]
    end

    γ = Nullable{T}()
    λ = Nullable(one(T)/3)
    W_k = MOD.qda!(copy(X_tmp), copy(M_tmp), y, γ, λ)
    W_k_tmp = MOD.class_whiteners!(copy(H_tmp), y, γ, get(λ))
    for i in eachindex(W_k)
        @test_approx_eq W_k[i] W_k_tmp[i]
    end

    γ = Nullable{T}(one(T)/3)
    λ = Nullable(one(T)/3)
    W_k = MOD.qda!(copy(X_tmp), copy(M_tmp), y, γ, λ)
    W_k_tmp = MOD.class_whiteners!(copy(H_tmp), y, γ, get(λ))
    for i in eachindex(W_k)
        @test_approx_eq W_k[i] W_k_tmp[i]
    end

end

info("Testing ", MOD.qda)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    M_tmp = convert(Matrix{T}, M)

    γ = Nullable{T}()
    λ = Nullable{T}()
    W_k_tmp = MOD.qda!(copy(X_tmp), M_tmp, y, γ, λ)
    model = qda(X_tmp, y)
    for i in eachindex(W_k_tmp)
        @test_approx_eq model.W_k[i] W_k_tmp[i]
    end

    γ = Nullable(one(T)/3)
    λ = Nullable{T}()
    W_k_tmp = MOD.qda!(copy(X_tmp), M_tmp, y, γ, λ)
    model = qda(X_tmp, y, gamma=γ)
    for i in eachindex(W_k_tmp)
        @test_approx_eq model.W_k[i] W_k_tmp[i]
    end

    γ = Nullable{T}()
    λ = Nullable(one(T)/3)
    W_k_tmp = MOD.qda!(copy(X_tmp), M_tmp, y, γ, λ)
    model = qda(X_tmp, y, lambda=λ)
    for i in eachindex(W_k_tmp)
        @test_approx_eq model.W_k[i] W_k_tmp[i]
    end

    γ = Nullable(one(T)/3)
    λ = Nullable(one(T)/3)
    W_k_tmp = MOD.qda!(copy(X_tmp), M_tmp, y, γ, λ)
    model = qda(X_tmp, y, lambda=λ, gamma=γ)
    for i in eachindex(W_k_tmp)
        @test_approx_eq model.W_k[i] W_k_tmp[i]
    end
end

info("Testing ", MOD.discriminants_qda)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    priors_tmp = convert(Vector{T}, priors)

    model = qda(X_tmp, y)
    Z2 = hcat([MOD.dotrows((X_tmp .- M[i,:])*model.W_k[i]) for i in eachindex(priors_tmp)]...)
    δ = -Z2/2 .+ log(priors_tmp)'
    @test_approx_eq δ MOD.discriminants(model, X_tmp)

    for γ in (zero(T), convert(T, 0.25), convert(T, 0.75), one(T))
        model = qda(X_tmp, y)
        Z2 = hcat([MOD.dotrows((X_tmp .- M[i,:])*model.W_k[i]) for i in eachindex(priors_tmp)]...)
        δ = -Z2/2 .+ log(priors_tmp)'
        @test_approx_eq δ MOD.discriminants(model, X_tmp)
    end
end
