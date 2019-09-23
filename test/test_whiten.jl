@info "Testing whiten.jl"

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