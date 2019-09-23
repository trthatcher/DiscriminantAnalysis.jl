@info "Testing common.jl"

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


# Data Validation

@testset "validate_priors(π)" begin
    k = 10
    for T in (Float32, Float64)
        # Check probability domain
        @test_throws DomainError DA.validate_priors(T[0.5; 0.5; 0.0])
        @test_throws DomainError DA.validate_priors(T[1.0; 0.0; 0.0])

        # Check totals
        @test_throws ArgumentError DA.validate_priors(T[0.3; 0.3; 0.3])
        @test_throws ArgumentError DA.validate_priors(T[0.4; 0.4; 0.4])

        # Test valid π
        π = rand(T, k)
        π ./= sum(π)
        #broadcast!(/, π, π, sum(π))

        @test k == DA.validate_priors(π)
    end
end

# Class operations

@testset "class_counts(nₘ, y)" begin
    m = 3
    nₖ = 300

    y = repeat(vec(1:m), outer=nₖ)

    # Test bounds
    y_tst = copy(y)

    y_tst[1] = 0
    @test_throws BoundsError DA.class_counts!(Vector{Int}(undef, m), y_tst)

    y_tst[1] = m + 1
    @test_throws BoundsError DA.class_counts!(Vector{Int}(undef, m), y_tst)

    # Test computation
    @test DA.class_counts!(Vector{Int}(undef, m), y) == [nₖ for i = 1:m]
end

@testset "_class_statistics!(M, nₘ, X, y)" begin
    nₘ = [45, 55, 100]
    m = length(nₘ)
    n = sum(nₘ)
    p = 5

    for T in (Float32, Float64)
        X, y, M = random_data(T, nₘ, p)
        Xtt = transpose(copy(transpose(X)))
        Mtt = transpose(copy(transpose(M)))

        for X_test in (X, Xtt)
            nₘ_test = copy(nₘ)
            # test predictor dimensionality
            @test_throws DimensionMismatch DA._class_statistics!(zeros(T,2,p+1), nₘ_test, X_test,  y)
            @test_throws DimensionMismatch DA._class_statistics!(zeros(T,2,p-1), nₘ_test, X_test,  y)

            # test observation dimensionality
            @test_throws DimensionMismatch DA._class_statistics!(similar(M), nₘ_test, X_test, zeros(Int,n+1))
            @test_throws DimensionMismatch DA._class_statistics!(similar(M), nₘ_test, X_test, zeros(Int,n-1))

            # test count dimensionality
            @test_throws DimensionMismatch DA._class_statistics!(similar(M), zeros(Int,m+1), X_test, y)
            @test_throws DimensionMismatch DA._class_statistics!(similar(M), zeros(Int,m-1), X_test, y)

            # test indexing of class_statistics
            y_test = copy(y)
            y_test[1] = 0
            @test_throws BoundsError DA._class_statistics!(similar(M), nₘ_test, X_test, y_test)

            y_test[1] = m+1
            @test_throws BoundsError DA._class_statistics!(similar(M), nₘ_test, X_test, y_test)
        end

        # test mean computation
        for (M_test, X_test) in ((copy(M), X), (deepcopy(Mtt), Xtt))
            nₘ_test = zeros(Int, m)

            M_res, nₘ_res = DA._class_statistics!(M_test, nₘ_test, X, y)
            
            @test nₘ_res === nₘ_test
            @test nₘ_res == nₘ

            @test M_res === M_test
            @test isapprox(M, M_test)
        end
    end
end

### Regularization

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