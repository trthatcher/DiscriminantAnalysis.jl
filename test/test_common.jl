@testset "class_means!(M, X, y)" begin
    n = 10
    p = 3
    ix = sortperm(rand(2n))
    y = repeat(1:2, inner=n)[ix]

    for T in (Float32, Float64)
        X1 = rand(T, n, p)
        X1 .-= mean(X1, dims=1)
        X1 .+= 1
        X2 = rand(T, n, p)
        X2 .-= mean(X2, dims=1)
        X2 .-= 1
        
        X = [X1; X2][ix, :]
        
        # test predictor dimensionality
        #print("y: ", y, "\n")
        #print("X: ", X, "\n")
        #DA.class_means!(zeros(T,2,p+1), X, y)
        @test_throws DimensionMismatch DA.class_means!(zeros(T,2,p+1), X, y)
        @test_throws DimensionMismatch DA.class_means!(zeros(T,2,p-1), X, y)

        # test observation dimensionality
        @test_throws DimensionMismatch DA.class_means!(zeros(T,2,p), X, zeros(Int,n+1))
        @test_throws DimensionMismatch DA.class_means!(zeros(T,2,p), X, zeros(Int,n-1))

        # test indexing of class_means
        y_test = copy(y)

        y_test[p] = 3
        @test_throws BoundsError DA.class_means!(zeros(T,2,p), X, y_test)

        y_test[p] = 0
        @test_throws BoundsError DA.class_means!(zeros(T,2,p), X, y_test)

        # test observation count
        y_test = copy(y)
        y_test .= 1

        @test_throws ErrorException DA.class_means!(zeros(T,2,p), X, y_test)

        # test mean computation
        M = zeros(T, 2, p)

        M_test = ones(T, 2, p)
        M_test[2, :] .= -1

        M_res = DA.class_means!(M, X, y)

        @test M === M_res
        @test isapprox(M, M_test)
    end
end

#@testset "_center_classes" begin
#    n = 10
#    p = 3
#    for T in (Float32, Float64), U in (Int32, Int64)
#        X1 = rand(n, p)
#        X2 = rand(n, p)
#
#    end
#end


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