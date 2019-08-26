# Dimensionality Checks

@testset "check_dims(X; dims)" begin
    n = 20
    p = 5
    for T in (Float32, Float64)
        @test_throws ArgumentError DA.check_dims(zeros(T, p, p), dims=0)
        @test_throws ArgumentError DA.check_dims(zeros(T, p, p), dims=3)

        @test (n, p) == DA.check_dims(zeros(T, n, p), dims=1)
        @test (n, p) == DA.check_dims(zeros(T, p, n), dims=2)
        @test (n, p) == DA.check_dims(transpose(zeros(T, n, p)), dims=2)
    end
end

@testset "check_centroid_dims(M, X; dims)" begin
    n = 20
    p = 5
    k = 3
    for T in (Float32, Float64)
        X = zeros(T, n, p)
        M = zeros(T, k, p)

        @test_throws ArgumentError DA.check_centroid_dims(M, X, dims=0)
        @test_throws ArgumentError DA.check_centroid_dims(M, X, dims=3)

        # check parameter dimensionality for row-based data

        @test_throws DimensionMismatch DA.check_centroid_dims(zeros(T, k, p+1), X, dims=1)
        @test_throws DimensionMismatch DA.check_centroid_dims(zeros(T, k, p-1), X, dims=1)

        @test_throws DimensionMismatch DA.check_centroid_dims(M, zeros(T, n, p+1), dims=1)
        @test_throws DimensionMismatch DA.check_centroid_dims(M, zeros(T, n, p-1), dims=1)

        @test (n, p, k) == DA.check_centroid_dims(M, X, dims=1)

        # check parameter dimensionality for column-based data

        Xt = transpose(X)
        Mt = transpose(M)

        @test_throws DimensionMismatch DA.check_centroid_dims(zeros(T, p+1, k), Xt, dims=2)
        @test_throws DimensionMismatch DA.check_centroid_dims(zeros(T, p-1, k), Xt, dims=2)

        @test_throws DimensionMismatch DA.check_centroid_dims(Mt, zeros(T, p+1, n), dims=2)
        @test_throws DimensionMismatch DA.check_centroid_dims(Mt, zeros(T, p-1, n), dims=2)

        @test (n, p, k) == DA.check_centroid_dims(Mt, Xt, dims=2)
    end
end

@testset "check_centroid_dims(M, π; dims)" begin
    k = 3
    p = 5
    for T in (Float32, Float64)
        M = zeros(T, k, p)
        π = zeros(T, k)

        @test_throws ArgumentError DA.check_centroid_dims(M, π, dims=0)
        @test_throws ArgumentError DA.check_centroid_dims(M, π, dims=3)

        # check parameter dimensionality for row-based data

        @test_throws DimensionMismatch DA.check_centroid_dims(zeros(T, k+1, p), π, dims=1)
        @test_throws DimensionMismatch DA.check_centroid_dims(zeros(T, k-1, p), π, dims=1)

        @test_throws DimensionMismatch DA.check_centroid_dims(M, zeros(T, k+1), dims=1)
        @test_throws DimensionMismatch DA.check_centroid_dims(M, zeros(T, k-1), dims=1)

        @test (k, p) == DA.check_centroid_dims(M, π, dims=1)

        # check parameter dimensionality for column-based data

        Mt = transpose(M)

        @test_throws DimensionMismatch DA.check_centroid_dims(zeros(T, k+1, p), π, dims=2)
        @test_throws DimensionMismatch DA.check_centroid_dims(zeros(T, k-1, p), π, dims=2)

        @test_throws DimensionMismatch DA.check_centroid_dims(Mt, zeros(T, k+1), dims=2)
        @test_throws DimensionMismatch DA.check_centroid_dims(Mt, zeros(T, k-1), dims=2)

        @test (k, p) == DA.check_centroid_dims(Mt, π, dims=2)
    end
end

@testset "check_data_dims(X, y; dims)" begin
    n = 10
    p = 5
    for T in (Float32, Float64)
        X = zeros(T, n, p)
        y = zeros(Int, n)

        @test_throws ArgumentError DA.check_data_dims(X, y, dims=0)
        @test_throws ArgumentError DA.check_data_dims(X, y, dims=3)

        # check parameter dimensionality for row-based data

        @test_throws DimensionMismatch DA.check_data_dims(zeros(T, n+1, p), y, dims=1)
        @test_throws DimensionMismatch DA.check_data_dims(zeros(T, n-1, p), y, dims=1)

        @test_throws DimensionMismatch DA.check_data_dims(X, zeros(Int, n+1), dims=1)
        @test_throws DimensionMismatch DA.check_data_dims(X, zeros(Int, n-1), dims=1)

        @test (n, p) == DA.check_data_dims(X, y, dims=1)

        # check parameter dimensionality for column-based data

        Xt = transpose(X)

        @test_throws DimensionMismatch DA.check_data_dims(zeros(T, p, n+1), y, dims=2)
        @test_throws DimensionMismatch DA.check_data_dims(zeros(T, p, n-1), y, dims=2)

        @test_throws DimensionMismatch DA.check_data_dims(Xt, zeros(Int, n+1), dims=2)
        @test_throws DimensionMismatch DA.check_data_dims(Xt, zeros(Int, n-1), dims=2)

        @test (n, p) == DA.check_data_dims(Xt, y, dims=2)
    end
end

@testset "check_priors(X, π)" begin
    k = 10
    for T in (Float32, Float64)
        # Check totals
        @test_throws ArgumentError DA.check_priors(T[0.3; 0.3; 0.3])
        @test_throws ArgumentError DA.check_priors(T[0.4; 0.4; 0.4])

        # Check probability domain
        @test_throws DomainError DA.check_priors(T[1.0; -0.5; 0.5])

        # Test valid π
        π = rand(T, k)
        broadcast!(/, π, π, sum(π))

        @test k == DA.check_priors(π)
    end
end

# Class operations

@testset "class_counts(y; m])" begin
    nₘ = [45, 55, 50]
    m = length(nₘ)
    n = sum(nₘ)
    p = 3

    for T in (Float32, Float64)
        y = random_data(T, nₘ, p)[2]

        # Test bounds
        y_tst = copy(y)

        y_tst[nₘ[1]] = 0
        @test_throws BoundsError DA.class_counts(y_tst, m=m)

        y_tst[nₘ[1]] = m + 1
        @test_throws BoundsError DA.class_counts(y_tst, m=m)

        # Test computation
        @test DA.class_counts(y) == nₘ
    end
end

@testset "class_means!(M, X, y)" begin
    nₖ = [45, 55]
    n = sum(nₖ)
    p = 3

    for T in (Float32, Float64)
        X, y, M = random_data(T, nₖ, p)
        Xt = transpose(copy(transpose(X)))
        Mt = transpose(copy(transpose(M)))
        
        # test predictor dimensionality
        @test_throws DimensionMismatch DA.class_means!(zeros(T,2,p+1), X,  y)
        @test_throws DimensionMismatch DA.class_means!(zeros(T,2,p-1), X,  y)
        @test_throws DimensionMismatch DA.class_means!(zeros(T,2,p+1), Xt, y)
        @test_throws DimensionMismatch DA.class_means!(zeros(T,2,p-1), Xt, y)

        # test observation dimensionality
        @test_throws DimensionMismatch DA.class_means!(similar(M),  X,  zeros(Int,n+1))
        @test_throws DimensionMismatch DA.class_means!(similar(M),  X,  zeros(Int,n-1))
        @test_throws DimensionMismatch DA.class_means!(similar(Mt), Xt, zeros(Int,n+1))
        @test_throws DimensionMismatch DA.class_means!(similar(Mt), Xt, zeros(Int,n-1))

        # test indexing of class_means
        y_test = copy(y)

        y_test[p] = 3
        @test_throws BoundsError DA.class_means!(similar(M),  X,  y_test)
        @test_throws BoundsError DA.class_means!(similar(Mt), Xt, y_test)

        y_test[p] = 0
        @test_throws BoundsError DA.class_means!(similar(M),  X,  y_test)
        @test_throws BoundsError DA.class_means!(similar(Mt), Xt, y_test)

        # test observation count
        y_test = copy(y)
        y_test .= 2

        @test_throws ErrorException DA.class_means!(similar(M),  X,  y_test)
        @test_throws ErrorException DA.class_means!(similar(Mt), Xt, y_test)

        # test mean computation
        M_test = similar(M)
        M_res = DA.class_means!(M_test, X, y)
        @test M_res === M_test
        @test isapprox(M, M_test)

        Mt_test = transpose(similar(transpose(M)))
        Mt_res = DA.class_means!(Mt_test, Xt, y)
        @test Mt_res === Mt_test
        @test isapprox(Mt, Mt_test)
    end
end

@testset "class_means(X, y; dims, m)" begin
    nₖ = [45, 55]
    n = sum(nₖ)
    p = 3

    for T in (Float32, Float64)
        X, y, M = random_data(T, nₖ, p)
        Xt = copy(transpose(X))

        # Check dims argument
        @test_throws ArgumentError DA.class_means(X, y, dims=0)
        @test_throws ArgumentError DA.class_means(X, y, dims=3)

        # Test predictor dimensionality
        @test_throws DimensionMismatch DA.class_means(X,  zeros(Int,n+1))
        @test_throws DimensionMismatch DA.class_means(X,  zeros(Int,n-1))

        @test_throws DimensionMismatch DA.class_means(Xt, zeros(Int,n+1), dims=2)
        @test_throws DimensionMismatch DA.class_means(Xt, zeros(Int,n-1), dims=2)

        # test indexing of class_means - careful of k argument
        y_test = copy(y)

        y_test[p] = 3
        @test_throws BoundsError DA.class_means(X,  y_test, dims=1, m=2)
        @test_throws BoundsError DA.class_means(Xt, y_test, dims=2, m=2)

        y_test[p] = 0
        @test_throws BoundsError DA.class_means(X,  y_test)
        @test_throws BoundsError DA.class_means(Xt, y_test, dims=2)

        # test observation count
        y_test = copy(y)
        y_test .= 2

        @test_throws ErrorException DA.class_means(X,  y_test)
        @test_throws ErrorException DA.class_means(Xt, y_test, dims=2)

        # test mean computation
        M_test = DA.class_means(X, y)
        @test isapprox(M_test, M)

        M_test = DA.class_means(Xt, y, dims=2)
        @test isapprox(M_test, transpose(M))
    end
end

@testset "_center_classes!(X, y, M)" begin
    nₖ = [45, 55]
    n = sum(nₖ)
    p = 3

    for T in (Float32, Float64)
        X, y, M = random_data(T, nₖ, p)
        Xtt = transpose(copy(transpose(X)))
        Mtt = transpose(copy(transpose(M)))

        # test predictor dimensionality
        @test_throws DimensionMismatch DA._center_classes!(X,  y, zeros(T,2,p+1))
        @test_throws DimensionMismatch DA._center_classes!(X,  y, zeros(T,2,p-1))
        @test_throws DimensionMismatch DA._center_classes!(Xtt, y, zeros(T,2,p+1))
        @test_throws DimensionMismatch DA._center_classes!(Xtt, y, zeros(T,2,p-1))

        # test observation dimensionality
        @test_throws DimensionMismatch DA._center_classes!(X,  zeros(Int,n+1), similar(M))
        @test_throws DimensionMismatch DA._center_classes!(X,  zeros(Int,n-1), similar(M))
        @test_throws DimensionMismatch DA._center_classes!(Xtt, zeros(Int,n+1), similar(Mtt))
        @test_throws DimensionMismatch DA._center_classes!(Xtt, zeros(Int,n-1), similar(Mtt))

        # test indexing of class_means - careful of k argument
        y_test = copy(y)

        y_test[p] = 3
        @test_throws BoundsError DA._center_classes!(copy(X),  y_test, M)
        @test_throws BoundsError DA._center_classes!(copy(Xtt), y_test, Mtt)

        y_test[p] = 0
        @test_throws BoundsError DA._center_classes!(copy(X),  y_test, M)
        @test_throws BoundsError DA._center_classes!(copy(Xtt), y_test, Mtt)

        # test centering
        X_center = X .- M[y, :]

        X_test = DA._center_classes!(X, y, M)
        @test X === X_test
        @test isapprox(X_center, X)

        Xtt_test = DA._center_classes!(Xtt, y, Mtt)
        @test Xtt === Xtt_test
        @test isapprox(X_center, Xtt)
    end
end

@testset "center_classes!(X, y, M; dims)" begin
    nₖ = [45, 55]
    n = sum(nₖ)
    p = 3

    for T in (Float32, Float64)
        X, y, M = random_data(T, nₖ, p)
        Xt = copy(transpose(X))
        Mt = copy(transpose(M))

        # check dims argument
        @test_throws ArgumentError DA.center_classes!(X, y, M, dims=0)
        @test_throws ArgumentError DA.center_classes!(X, y, M, dims=3)

        # test predictor dimensionality
        @test_throws DimensionMismatch DA.center_classes!(X, y, zeros(T,2,p+1), dims=1)
        @test_throws DimensionMismatch DA.center_classes!(X, y, zeros(T,2,p-1), dims=1)

        @test_throws DimensionMismatch DA.center_classes!(Xt, y, zeros(T,p+1,2), dims=2)
        @test_throws DimensionMismatch DA.center_classes!(Xt, y, zeros(T,p-1,2), dims=2)

        # test observation dimensionality
        @test_throws DimensionMismatch DA.center_classes!(X, zeros(Int,n+1), M, dims=1)
        @test_throws DimensionMismatch DA.center_classes!(X, zeros(Int,n-1), M, dims=1)

        @test_throws DimensionMismatch DA.center_classes!(Xt, zeros(Int,n+1), Mt, dims=2)
        @test_throws DimensionMismatch DA.center_classes!(Xt, zeros(Int,n-1), Mt, dims=2)

        # test indexing of class_means - careful of k argument
        y_test = copy(y)

        y_test[p] = 3
        @test_throws BoundsError DA.center_classes!(copy(X),   y_test, M, dims=1)
        @test_throws BoundsError DA.center_classes!(copy(Xt), y_test, Mt, dims=2)

        y_test[p] = 0
        @test_throws BoundsError DA.center_classes!(copy(X),   y_test, M, dims=1)
        @test_throws BoundsError DA.center_classes!(copy(Xt), y_test, Mt, dims=2)

        # test centering
        X_center = X .- M[y, :]

        X_test = DA.center_classes!(X, y, M, dims=1)
        @test X === X_test
        @test isapprox(X_center, X)

        Xt_test = DA.center_classes!(Xt, y, Mt, dims=2)
        @test Xt === Xt_test
        @test isapprox(transpose(X_center), Xt)
    end
end


@testset "regularize!(Σ₁, Σ₂, λ)" begin
    for T in (Float32, Float64)
        S = zeros(T, 3, 3)
        @test_throws DimensionMismatch DA.regularize!(zeros(T, 2, 3), S, T(0))
        @test_throws DimensionMismatch DA.regularize!(S, zeros(T, 2, 3), T(0))
        @test_throws DimensionMismatch DA.regularize!(S, zeros(T, 2, 2), T(0))
        
        @test_throws DomainError DA.regularize!(S, S, T(0) - eps(T(0)))
        @test_throws DomainError DA.regularize!(S, S, T(1) + eps(T(1)))
        
        S1 = cov(rand(T, 10, 3))
        S2 = cov(rand(T, 10, 3))
        for λ in (T(0), T(0.25), T(0.5), T(0.75), T(1))
            S1_test = copy(S1)
            S2_test = copy(S2)

            DA.regularize!(S1_test, S2_test, λ)

            @test isapprox(S1_test, (1-λ)S1 + λ*S2)
            @test S2_test == S2
        end
    end
end

@testset "regularize!(Σ, γ)" begin
    n = 10
    p = 3
    for T in (Float32, Float64)
        @test_throws DimensionMismatch DA.regularize!(zeros(T, 2, 3), T(0))
        @test_throws DimensionMismatch DA.regularize!(zeros(T, 3, 2), T(0))
        
        S = zeros(T, p, p)
        @test_throws DomainError DA.regularize!(S, T(0) - eps(T(0)))
        @test_throws DomainError DA.regularize!(S, T(1) + eps(T(1)))

        S = cov(rand(T, n, p))
        for γ in (T(0), T(0.25), T(0.5), T(0.75), T(1))
            S_test = copy(S)

            DA.regularize!(S_test, γ)

            @test isapprox(S_test, (1-γ)S + γ*(tr(S)/p)I)
        end
    end
end

@testset "whiten_data!(X; dims, df)" begin
    n = 10
    p = 3
    for T in (Float32, Float64)
        # test matrix with too few rows
        @test_throws ErrorException DA.whiten_data!(zeros(T,p,p), dims=1)
        @test_throws ErrorException DA.whiten_data!(zeros(T,p,p), dims=2)
        
        # test singular matrix
        @test_throws ErrorException DA.whiten_data!(zeros(T,n,p), dims=1)
        @test_throws ErrorException DA.whiten_data!(zeros(T,p,n), dims=2)
        
        # test whitening 
        X = T[diagm(0 => ones(T, p));
              rand(n-p, p) .- 0.5]
        X .= X .- mean(X, dims=1)  # Data must be centered
        Xt = copy(transpose(X))

        ### rows
        X_test = copy(X)
        W, detΣ = DA.whiten_data!(X_test, dims=1)
        @test isapprox(cov(X*W, dims=1), diagm(0 => ones(T, p)))
        @test isapprox(det(cov(X)), detΣ)

        ### cols
        Xt_test = copy(Xt)
        W, detΣ = DA.whiten_data!(Xt_test, dims=2)
        @test isapprox(cov(W*Xt, dims=2), diagm(0 => ones(T, p)))
        @test isapprox(det(cov(X)), detΣ)
    end
end

@testset "whiten_data(X, γ, dims, ϵ)" begin
    n = 10
    p = 3
    for T in (Float32, Float64)
        # test limits for γ
        @test_throws ErrorException DA.whiten_data!(zeros(T,p,p), T(0) - eps(T(0)), dims=1)
        @test_throws ErrorException DA.whiten_data!(zeros(T,p,p), T(1) - eps(T(1)), dims=1)

        # test matrix with too few rows
        @test_throws ErrorException DA.whiten_data!(zeros(T,p,p), T(0), dims=1)
        @test_throws ErrorException DA.whiten_data!(zeros(T,p,p), T(0), dims=2)
        
        # test singular matrix
        @test_throws ErrorException DA.whiten_data!(zeros(T,n,p), T(0), dims=1)
        @test_throws ErrorException DA.whiten_data!(zeros(T,p,n), T(0), dims=2)
        
        # test whitening 
        X = T[diagm(0 => ones(T, p));
              rand(n-p, p) .- 0.5]
        X .= X .- mean(X, dims=1)  # Data must be centered
        Xt = copy(transpose(X))

        ### rows
        X_test = copy(X)
        W = DA.whiten_data!(X_test, T(0), dims=1)
        @test isapprox(cov(X*W, dims=1), diagm(0 => ones(T, p)))

        ### cols
        Xt_test = copy(Xt)
        W = DA.whiten_data!(Xt_test, T(0), dims=2)
        @test isapprox(cov(W*Xt, dims=2), diagm(0 => ones(T, p)))
    end
end