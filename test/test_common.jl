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

n_k = [4; 7; 5]
k = length(n_k)
p = 4
y = MOD.RefVector(vcat([Int64[i for j = 1:n_k[i]] for i = 1:k]...), k)
X = vcat([rand(n_k[i], p) .+ (10rand(1,p) .- 5) for i = 1:k]...)
σ = sortperm(rand(sum(n_k)))
y = MOD.RefVector(y[σ])
X = X[σ,:]

info("Testing ", MOD.classcounts)
for U in IntegerTypes
    @test all(n_k .== MOD.classcounts(convert(MOD.RefVector{U}, y)))
end

info("Testing ", MOD.classtotals)
for T in FloatingPointTypes
    for U in IntegerTypes
        X_tmp = convert(Array{T}, X)
        y_tmp = convert(MOD.RefVector{U}, y)
        test_ans = vcat([sum(X_tmp[y.ref .== i,:],1) for i = 1:k]...)
        @test_approx_eq MOD.classtotals(Val{:row}, X_tmp, y_tmp)  test_ans
        @test_approx_eq MOD.classtotals(Val{:col}, X_tmp', y_tmp) test_ans'
        @test_throws DimensionMismatch MOD.classtotals(Val{:row}, X_tmp', y_tmp)
        @test_throws DimensionMismatch MOD.classtotals(Val{:col}, X_tmp, y_tmp)
    end
end

info("Testing ", MOD.classmeans)
for T in FloatingPointTypes
    for U in IntegerTypes
        X_tmp = convert(Array{T}, X)
        y_tmp = convert(MOD.RefVector{U}, y)
        test_ans = MOD.classtotals(Val{:row}, X_tmp, y_tmp) ./ n_k
        @test_approx_eq MOD.classmeans(Val{:row}, X_tmp,  y_tmp) test_ans
        @test_approx_eq MOD.classmeans(Val{:col}, X_tmp', y_tmp) test_ans'
    end
end

info("Testing ", MOD.centerclasses!)
for T in FloatingPointTypes
    for U in IntegerTypes
        X_tmp = copy(convert(Array{T}, X))
        y_tmp = convert(MOD.RefVector{U}, y)
        M_tmp = MOD.classmeans(Val{:row}, X_tmp, y_tmp)
        test_ans = X_tmp .- M_tmp[y_tmp, :]
        @test_approx_eq MOD.centerclasses!(Val{:row}, copy(X_tmp), M_tmp, y_tmp) test_ans
        @test_approx_eq MOD.centerclasses!(Val{:col}, copy(X_tmp)', M_tmp', y_tmp) test_ans'
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
    A  = T[1 2 3;
           4 5 6;
           7 8 9;
           5 3 2]

    @test_approx_eq MOD.dotvectors(Val{:row}, A) sum(A .* A,2)
    @test_approx_eq MOD.dotvectors(Val{:col}, A) sum(A .* A,1)
end

#=
info("Testing ", MOD.whiten_data!)
for T in FloatingPointTypes
    X = T[1 0 0;
          0 1 0;
          0 0 1;
          5 5 3]
    μ = mean(X,1)
    H = X .- μ
    Σ = H'H/(size(X,1)-1)

    W = MOD.whiten_data!(copy(H), Nullable{T}())
    @test_approx_eq eye(T,3) cov(X*W)

    for λ in (convert(T, 0.25), convert(T, 0.5), convert(T, 0.75))
        W = MOD.whiten_data!(copy(H), Nullable(λ))
        @test_approx_eq eye(T,3) W'*((1-λ)*Σ + (λ*trace(Σ)/size(X,2))*I)*W
    end

    H = eye(T,3) .- mean(eye(T,3))
    @test_throws ErrorException MOD.whiten_data!(H, Nullable{T}())
end

info("Testing ", MOD.whiten_cov!)
for T in FloatingPointTypes
    X = T[1 0 0;
          0 1 0;
          0 0 1;
          5 5 3]
    μ = mean(X,1)
    H = X .- μ
    Σ = H'H/(size(X,1)-1)

    W = MOD.whiten_cov!(copy(Σ), Nullable{T}())
    @test_approx_eq eye(T,3) cov(X*W)

    for λ in (convert(T, 0.25), convert(T, 0.5), convert(T, 0.75))
        W = MOD.whiten_cov!(copy(Σ), Nullable(λ))
        @test_approx_eq eye(T,3) W'*((1-λ)*Σ + (λ*trace(Σ)/size(X,2))*I)*W
    end

    @test_throws ErrorException MOD.whiten_cov!(diagm([one(T); zero(T)]), Nullable{T}())
end
=#
