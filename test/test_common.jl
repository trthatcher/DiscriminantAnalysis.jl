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


### Model Parameters

@testset "parameter_fit!(Θ, y, X, dims, args...)" begin
    nₘ = [400; 500; 600]
    n = sum(nₘ)
    m = length(nₘ)
    p = 10

    for T in (Float32, Float64)
        X, y, M = random_data(T, nₘ, p)
        π = convert(Vector{T}, nₘ/n)

        M_tests = [nothing, perturb(M)]
        π_tests = [nothing, ones(T,m)/m]
        γ_tests = [nothing, range(zero(T), stop=one(T), length=3)...]

        DP = DA.DiscriminantParameters{T}

        for M_test in M_tests, π_test in π_tests, γ_test in γ_tests
            Xc = X .- (M_test === nothing ? M[:, y] : M_test[:, y])
            Σ = (Xc*transpose(Xc)) ./ (n-m)

            if !(γ_test === nothing)
                Σ = (1-γ_test)*Σ + γ_test*(tr(Σ)/p)*I
            end

            dp_test = DA.parameter_fit!(DP(), y, copy(X), 2, true, M_test, π_test, γ_test)

            @test dp_test.fit == false
            @test dp_test.dims == 2
            @test isapprox(dp_test.M, M_test === nothing ? M : M_test)
            @test isapprox(dp_test.π, π_test === nothing ? π : π_test)
            @test dp_test.nₘ == nₘ
            @test dp_test.γ == γ_test
            #@test isapprox(dp_test.detΣ, det(Σ))
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
        @test isapprox(det(cov(X, dims=1)), detΣ)

        ### cols
        Xt_test = copy(Xt)
        W, detΣ = DA.whiten_data!(Xt_test, dims=2)
        @test isapprox(cov(W*Xt, dims=2), diagm(0 => ones(T, p)))
        @test isapprox(det(cov(Xt, dims=2)), detΣ)
    end
end

@testset "whiten_data!(X, γ; dims, df)" begin
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
        Iₚ = diagm(0 => ones(T, p))

        X = T[Iₚ; rand(n-p, p) .- 0.5]
        X .= X .- mean(X, dims=1)  # Data must be centered
        Xt = copy(transpose(X))

        S = X'X
        Σ = S/(n-1)

        for γ in range(zero(T), step=convert(T, 0.25), stop=one(T))
            S_γ = (one(T) - γ)*S + γ*(tr(S)/p)*I
            Σ_γ = (one(T) - γ)*Σ + γ*(tr(Σ)/p)*I

            ### rows
            W_test, detΣ_test = DA.whiten_data!(copy(X), γ, dims=1)

            @test isapprox(det(Σ_γ), detΣ_test)
            @test isapprox(transpose(W_test)*Σ_γ*W_test, Iₚ)
            if γ == zero(T)
                @test isapprox(cov(X*W_test, dims=1), Iₚ)
            end

            W_test, detS_test = DA.whiten_data!(copy(X), γ, dims=1, df=1)
            @test isapprox(det(S_γ), detS_test)
            @test isapprox(transpose(W_test)*S_γ*W_test, Iₚ)

            ### cols
            W_test, detΣ_test = DA.whiten_data!(copy(Xt), γ, dims=2)

            @test isapprox(det(Σ_γ), detΣ_test)
            @test isapprox(W_test*Σ_γ*transpose(W_test), Iₚ)
            if γ == zero(T)
                @test isapprox(cov(W_test*Xt, dims=2), Iₚ)
            end

            W_test, detS_test = DA.whiten_data!(copy(Xt), γ, dims=2, df=1)
            @test isapprox(det(S_γ), detS_test)
            @test isapprox(W_test*S_γ*transpose(W_test), Iₚ)
        end
    end
end