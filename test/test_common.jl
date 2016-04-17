info("Testing ", MOD.RefVector)
for T in IntegerTypes
    ref = T[1; 1; 2; 3; 3]
    k = convert(T, 3)
    y = MOD.RefVector(ref,k)

    @test all(y.ref .== ref)
    @test y.k == k

    @test_throws ErrorException MOD.RefVector(ref[ref .!= 2], k)
    @test_throws ErrorException MOD.RefVector(ref[ref .!= 2], k + one(T))
    @test_throws ErrorException MOD.RefVector(ref[ref .!= 2], k - one(T))
end

# Class Functions

n_k = [80; 100; 120]
k = length(n_k)
n = sum(n_k)
p = 3

y, X = sampledata(n_k, p)
Z = vcat([sum(X[y.ref .== i,:],1) for i = 1:k]...)
M = Z ./ n_k
H = X .- M[y, :]
Σ = H'H/(n-1)

#=
y = MOD.RefVector(vcat([Int64[i for j = 1:n_k[i]] for i = 1:k]...), k)
X = vcat([rand(n_k[i], p) .+ (10rand(1,p) .- 5) for i = 1:k]...)
σ = sortperm(rand(sum(n_k)))
y = MOD.RefVector(y[σ])
X = X[σ,:]
=#

info("Testing ", MOD.classcounts)
for U in IntegerTypes
    @test all(n_k .== MOD.classcounts(convert(MOD.RefVector{U}, y)))
end

info("Testing ", MOD.classtotals)
for T in FloatingPointTypes
    for U in IntegerTypes
        X_tmp = convert(Array{T}, X)
        y_tmp = convert(MOD.RefVector{U}, y)
        Z_tmp = convert(Array{T}, Z)
        @test_approx_eq MOD.classtotals(Val{:row}, X_tmp, y_tmp)  Z_tmp
        @test_approx_eq MOD.classtotals(Val{:col}, X_tmp', y_tmp) Z_tmp'
        @test_throws DimensionMismatch MOD.classtotals(Val{:row}, X_tmp', y_tmp)
        @test_throws DimensionMismatch MOD.classtotals(Val{:col}, X_tmp, y_tmp)
    end
end

info("Testing ", MOD.classmeans)
for T in FloatingPointTypes
    for U in IntegerTypes
        X_tmp = convert(Array{T}, X)
        y_tmp = convert(MOD.RefVector{U}, y)
        M_tmp = convert(Array{T}, M)
        @test_approx_eq MOD.classmeans(Val{:row}, X_tmp,  y_tmp) M
        @test_approx_eq MOD.classmeans(Val{:col}, X_tmp', y_tmp) M'
    end
end

info("Testing ", MOD.centerclasses!)
for T in FloatingPointTypes
    for U in IntegerTypes
        X_tmp = copy(convert(Array{T}, X))
        y_tmp = convert(MOD.RefVector{U}, y)
        M_tmp = convert(Array{T}, M)
        H_tmp = convert(Array{T}, H)
        @test_approx_eq MOD.centerclasses!(Val{:row}, copy(X_tmp), M_tmp, y_tmp) H
        @test_approx_eq MOD.centerclasses!(Val{:col}, copy(X_tmp)', M_tmp', y_tmp) H'

        @test_throws DimensionMismatch MOD.centerclasses!(Val{:row}, X_tmp, Array(T,k,p+1), y_tmp)
        @test_throws DimensionMismatch MOD.centerclasses!(Val{:col}, X_tmp', Array(T,p+1,k), y_tmp)

        @test_throws DimensionMismatch MOD.centerclasses!(Val{:row}, X_tmp, Array(T,k+1,p), y_tmp)
        @test_throws DimensionMismatch MOD.centerclasses!(Val{:col}, X_tmp', Array(T,p,k+1), y_tmp)
    end
end


info("Testing ", MOD.regularize!)
for T in FloatingPointTypes
    S1 = T[1 2 3;
           4 5 6;
           7 8 9]

    n = size(S1,1)

    S2 = T[7 8 9;
           8 3 6;
           1 7 4]

    B = T[1 2;
          3 4;
          5 6]

    b = T[1;
          3]

    @test_approx_eq MOD.regularize!(copy(S1), zero(T),  S2) S1
    @test_approx_eq MOD.regularize!(copy(S1), one(T),   S2) S2
    @test_approx_eq MOD.regularize!(copy(S1), one(T)/2, S2) (1-one(T)/2)*S1 + (one(T)/2)*S2
    @test_throws ErrorException MOD.regularize!(copy(S1), -one(T),  S2)
    @test_throws ErrorException MOD.regularize!(copy(S1), 2*one(T), S2)
    @test_throws DimensionMismatch MOD.regularize!(S1, one(T), B)

    @test_approx_eq MOD.regularize!(copy(S1), zero(T)) S1
    @test_approx_eq MOD.regularize!(copy(S1), one(T)) (trace(S1)/n)*eye(n)
    @test_approx_eq MOD.regularize!(copy(S1), one(T)/2) (1-one(T)/2)*S1 + (one(T)/2)*(trace(S1)/n)*eye(n)
end

info("Testing ", MOD.dotvectors)
for T in FloatingPointTypes
    X_tmp = convert(Array{T}, X)

    @test_approx_eq MOD.dotvectors(Val{:row}, X_tmp)  sum(X_tmp .* X_tmp,2)
    @test_approx_eq MOD.dotvectors(Val{:col}, X_tmp') sum(X_tmp .* X_tmp,2)

    @test_throws DimensionMismatch MOD.dotvectors!(Val{:row}, X_tmp,  Array(T, n+1))
    @test_throws DimensionMismatch MOD.dotvectors!(Val{:col}, X_tmp', Array(T, n+1))
end

info("Testing ", MOD.whitendata_svd!)
for T in FloatingPointTypes
    H_tmp = copy(convert(Array{T}, H))
    Σ_tmp = copy(convert(Array{T}, Σ))

    # Test full rank case
    for λ in (zero(T), convert(T, 0.25), convert(T, 0.5), convert(T, 0.75), one(T))
        W_tmp = MOD.whitendata_svd!(copy(H_tmp), λ)
        S_tmp = (1-λ)*Σ_tmp + (λ*trace(Σ_tmp)/p)*I
        @test_approx_eq eye(T,3) W_tmp*S_tmp*(W_tmp')
    end

    # Test degenerate case
    H_tmp = eye(T,3) .- mean(eye(T,3))
    @test_throws ErrorException MOD.whitendata_svd!(copy(H_tmp), zero(T))
end

info("Testing ", MOD.whitendata_qr!)
for T in FloatingPointTypes
    H_tmp = copy(convert(Array{T}, H))
    Σ_tmp = copy(convert(Array{T}, Σ))

    # Test full rank case
    W_tmp = MOD.whitendata_qr!(copy(H_tmp))
    @test_approx_eq eye(T,3) (W_tmp')*Σ_tmp*W_tmp

    # Test degenerate case
    H_tmp = eye(T,3)
    @test_throws ErrorException MOD.whitendata_qr!(copy(H_tmp))

    H_tmp = ones(T,4,3)
    @test_throws ErrorException MOD.whitendata_qr!(copy(H_tmp))
end

info("Testing ", MOD.whitencov_chol!)
for T in FloatingPointTypes
    H_tmp = copy(convert(Array{T}, H))
    Σ_tmp = copy(convert(Array{T}, Σ))

    # Test unregularized, full rank case
    W_tmp = MOD.whitencov_chol!(copy(Σ_tmp), Nullable{T}())
    @test_approx_eq eye(T,3) (W_tmp')*Σ_tmp*W_tmp

    # Test regularized cases
    for λ in (zero(T), convert(T, 0.25), convert(T, 0.5), convert(T, 0.75), one(T))
        W_tmp = MOD.whitencov_chol!(copy(Σ_tmp), Nullable(λ))
        S_tmp = (1-λ)*Σ_tmp + (λ*trace(Σ_tmp)/p)*I
        @test_approx_eq eye(T,3) (W_tmp')*S_tmp*W_tmp
    end

    # Test degenerate case
    @test_throws ErrorException MOD.whitencov_chol!(diagm([one(T); zero(T)]), Nullable{T}())
    @test_throws ErrorException MOD.whitencov_chol!(diagm([one(T); zero(T)]), Nullable(zero(T)))
end
