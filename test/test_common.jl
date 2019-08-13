@testset "_class_means!(M, X, y)" begin
    n = 10
    p = 3

    for T in (Float32, Float64)
        X, y, M = generate_data(T, n, p)
        Xt = transpose(copy(transpose(X)))
        Mt = transpose(copy(transpose(M)))
        
        # test predictor dimensionality
        @test_throws DimensionMismatch DA._class_means!(zeros(T,2,p+1), X,  y)
        @test_throws DimensionMismatch DA._class_means!(zeros(T,2,p-1), X,  y)
        @test_throws DimensionMismatch DA._class_means!(zeros(T,2,p+1), Xt, y)
        @test_throws DimensionMismatch DA._class_means!(zeros(T,2,p-1), Xt, y)

        # test observation dimensionality
        @test_throws DimensionMismatch DA._class_means!(similar(M),  X,  zeros(Int,n+1))
        @test_throws DimensionMismatch DA._class_means!(similar(M),  X,  zeros(Int,n-1))
        @test_throws DimensionMismatch DA._class_means!(similar(Mt), Xt, zeros(Int,n+1))
        @test_throws DimensionMismatch DA._class_means!(similar(Mt), Xt, zeros(Int,n-1))

        # test indexing of class_means
        y_test = copy(y)

        y_test[p] = 3
        @test_throws BoundsError DA._class_means!(similar(M),  X,  y_test)
        @test_throws BoundsError DA._class_means!(similar(Mt), Xt, y_test)

        y_test[p] = 0
        @test_throws BoundsError DA._class_means!(similar(M),  X,  y_test)
        @test_throws BoundsError DA._class_means!(similar(Mt), Xt, y_test)

        # test observation count
        y_test = copy(y)
        y_test .= 2

        @test_throws ErrorException DA._class_means!(similar(M),  X,  y_test)
        @test_throws ErrorException DA._class_means!(similar(Mt), Xt, y_test)

        # test mean computation
        M_test = similar(M)
        M_res = DA._class_means!(M_test, X, y)
        @test M_test === M_res
        @test isapprox(M, M_test)

        Mt_test = transpose(similar(transpose(M)))
        M_res = DA._class_means!(Mt_test, Xt, y)
        @test Mt_test === M_res
        @test isapprox(M, M_test)
    end
end

@testset "_class_means(X, y; dims, k)" begin
    n = 10
    p = 3

    for T in (Float32, Float64)
        X, y, M = generate_data(T, n, p)
        Xt = copy(transpose(X))

        # Check dims argument
        @test_throws ArgumentError DA._class_means(X, y, dims=0)
        @test_throws ArgumentError DA._class_means(X, y, dims=3)

        # Test predictor dimensionality
        @test_throws DimensionMismatch DA._class_means(X,  zeros(Int,n+1))
        @test_throws DimensionMismatch DA._class_means(X,  zeros(Int,n-1))

        @test_throws DimensionMismatch DA._class_means(Xt, zeros(Int,n+1), dims=2)
        @test_throws DimensionMismatch DA._class_means(Xt, zeros(Int,n-1), dims=2)

        # test indexing of class_means - careful of k argument
        y_test = copy(y)

        y_test[p] = 3
        @test_throws BoundsError DA._class_means(X,  y_test, k=2)
        @test_throws BoundsError DA._class_means(Xt, y_test, dims=2, k=2)

        y_test[p] = 0
        @test_throws BoundsError DA._class_means(X,  y_test)
        @test_throws BoundsError DA._class_means(Xt, y_test, dims=2)

        # test observation count
        y_test = copy(y)
        y_test .= 2

        @test_throws ErrorException DA._class_means(X,  y_test)
        @test_throws ErrorException DA._class_means(Xt, y_test, dims=2)

        # test mean computation
        M_test = DA._class_means(X, y)
        @test isapprox(M, M_test)

        M_test = DA._class_means(Xt, y, dims=2)
        @test isapprox(transpose(M), M_test)
    end
end

@testset "_center_classes!(X, y, M)" begin
    n = 10
    p = 3

    for T in (Float32, Float64)
        X, y, M = generate_data(T, n, p)
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

@testset "_center_classes!(X, y, M, dims)" begin
    n = 10
    p = 3

    for T in (Float32, Float64)
        X, y, M = generate_data(T, n, p)
        Xt = copy(transpose(X))
        Mt = copy(transpose(M))

        # check dims argument
        @test_throws ArgumentError DA._center_classes!(X, y, M, 0)
        @test_throws ArgumentError DA._center_classes!(X, y, M, 3)

        # test predictor dimensionality
        @test_throws DimensionMismatch DA._center_classes!(X, y, zeros(T,2,p+1), 1)
        @test_throws DimensionMismatch DA._center_classes!(X, y, zeros(T,2,p-1), 1)

        @test_throws DimensionMismatch DA._center_classes!(Xt, y, zeros(T,p+1,2), 2)
        @test_throws DimensionMismatch DA._center_classes!(Xt, y, zeros(T,p-1,2), 2)

        # test observation dimensionality
        @test_throws DimensionMismatch DA._center_classes!(X, zeros(Int,n+1), M, 1)
        @test_throws DimensionMismatch DA._center_classes!(X, zeros(Int,n-1), M, 1)

        @test_throws DimensionMismatch DA._center_classes!(Xt, zeros(Int,n+1), Mt, 2)
        @test_throws DimensionMismatch DA._center_classes!(Xt, zeros(Int,n-1), Mt, 2)

        # test indexing of class_means - careful of k argument
        y_test = copy(y)

        y_test[p] = 3
        @test_throws BoundsError DA._center_classes!(copy(X),   y_test, M, 1)
        @test_throws BoundsError DA._center_classes!(copy(Xt), y_test, Mt, 2)

        y_test[p] = 0
        @test_throws BoundsError DA._center_classes!(copy(X),   y_test, M, 1)
        @test_throws BoundsError DA._center_classes!(copy(Xt), y_test, Mt, 2)

        # test centering
        X_center = X .- M[y, :]

        X_test = DA._center_classes!(X, y, M, 1)
        @test X === X_test
        @test isapprox(X_center, X)

        Xt_test = DA._center_classes!(Xt, y, Mt, 2)
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

@testset "_whiten_data(X)" begin
    n = 10
    p = 3
    for T in (Float32, Float64)
        # test matrix with too few rows
        @test_throws ErrorException DA._whiten_data!(zeros(T,p,p), 1)
        @test_throws ErrorException DA._whiten_data!(zeros(T,p,p), 2)
        
        # test singular matrix
        @test_throws ErrorException DA._whiten_data!(zeros(T,n,p), 1)
        @test_throws ErrorException DA._whiten_data!(zeros(T,p,n), 2)
        
        # test whitening 
        X = T[diagm(0 => ones(T, p));
              rand(n-p, p) .- 0.5]
        X .= X .- mean(X, dims=1)  # Data must be centered
        Xt = copy(transpose(X))

        ### rows
        X_test = copy(X)
        W = DA._whiten_data!(X_test, 1)
        @test isapprox(cov(X*W, dims=1), diagm(0 => ones(T, p)))

        ### cols
        Xt_test = copy(Xt)
        W = DA._whiten_data!(Xt_test, 2)
        @test isapprox(cov(W*Xt, dims=2), diagm(0 => ones(T, p)))
    end
end