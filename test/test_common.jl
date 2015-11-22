info("Testing ", MOD.translate!)
for T in FloatingPointTypes
    A = T[1 2;
          3 4;
          5 6]

    b = T[1;
          2]

    c = T[1;
          2;
          3]

    @test_approx_eq MOD.translate!(copy(A), one(T)) (A .+ one(T))
    @test_approx_eq MOD.translate!(one(T), copy(A)) (A .+ one(T))
    @test_approx_eq MOD.translate!(copy(A), b)      (A .+ b')
    @test_approx_eq MOD.translate!(c, copy(A))      (A .+ c)
end

info("Testing ", MOD.regularize!)
for T in FloatingPointTypes
    S1 = T[1 2 3;
           4 5 6;
           7 8 9]

    S2 = T[7 8 9;
           8 3 6;
           1 7 4]

    s2 = T[5;
           7;
           9]

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

    @test_approx_eq MOD.regularize!(copy(S1), zero(T), s2)  S1
    @test_approx_eq MOD.regularize!(copy(S1), one(T),  s2)  diagm(s2)
    @test_approx_eq MOD.regularize!(copy(S1), one(T)/2, s2) (1-one(T)/2)*S1 + (one(T)/2)*diagm(s2)

    @test_throws ErrorException MOD.regularize!(copy(S1), -one(T),  s2)
    @test_throws ErrorException MOD.regularize!(copy(S1), 2*one(T), s2)

    @test_throws DimensionMismatch MOD.regularize!(S1, one(T), b)
end

info("Testing ", MOD.symml)
for T in FloatingPointTypes
    A  = T[1 2 3;
           4 5 6;
           7 8 9]
    AL = T[1 2 3;
           2 5 6;
           3 6 9]
    AU = T[1 4 7;
           4 5 8;
           7 8 9]

    B = MOD.symml(A)
    @test eltype(B) == T
    @test_approx_eq B AL
end


info("Testing ", MOD.dot_rows)
for T in FloatingPointTypes
    A  = T[1 2 3;
           4 5 6;
           7 8 9;
           5 3 2]

    @test_approx_eq MOD.dot_rows(A) sum(A .* A,2)
end

info("Testing ", MOD.dot_columns)
for T in FloatingPointTypes
    A  = T[1 2 3;
           4 5 6;
           7 8 9;
           5 3 2]

    @test_approx_eq MOD.dot_columns(A) sum(A .* A,1)
end


# Class Functions

n_k = [4; 7; 5]
k = length(n_k)
p = 4
y = vcat([Int64[i for j = 1:n_k[i]] for i = 1:k]...)
X = vcat([rand(n_k[i], p) .+ (10rand(1,p) .- 5) for i = 1:k]...)
σ = sortperm(rand(sum(n_k)))
y = y[σ]
X = X[σ,:]

info("Testing ", MOD.class_counts)
for U in IntegerTypes
    @test all(n_k .== MOD.class_counts(convert(Array{U}, y)))
end

info("Testing ", MOD.class_totals)
for T in FloatingPointTypes
    for U in IntegerTypes
        X_tmp = convert(Array{T}, X)
        y_tmp = convert(Array{U}, y)
        @test_approx_eq MOD.class_totals(X_tmp, y_tmp) vcat([sum(X_tmp[y .== i,:],1) for i = 1:k]...)
    end
end

info("Testing ", MOD.class_means)
for T in FloatingPointTypes
    for U in IntegerTypes
        X_tmp = convert(Array{T}, X)
        y_tmp = convert(Array{U}, y)        
        @test_approx_eq MOD.class_means(X_tmp, y_tmp) (MOD.class_totals(X_tmp, y_tmp) ./ n_k)
    end
end

info("Testing ", MOD.center_classes!)
for T in FloatingPointTypes
    for U in IntegerTypes
        X_tmp = copy(convert(Array{T}, X))
        y_tmp = copy(convert(Array{U}, y))
        M = MOD.class_means(X_tmp, y_tmp)
        @test_approx_eq MOD.center_classes!(copy(X_tmp), M, y_tmp) (X_tmp .- M[y_tmp, :])
    end
end
