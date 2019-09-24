@info "Testing linear.jl"

@testset "canonical_coordinates!(LDA)" begin
    p = 10
    m = 5

    for T in (Float32, Float64)
        LDM = DA.LinearDiscriminantModel{T}

        Σ_within, W_svd, W_chol = random_cov(T, p)

        π = rand(T, m)
        π ./= sum(π)

        M = random_centroids(T, m, p)
        μ = vec(M*π)
        Mc = M .- μ

        Σ_between = Symmetric(Mc*(π .* transpose(Mc)))
        C_ref = eigen(Σ_between, Σ_within).vectors[:, range(p, step=-1, length=m-1)]
        C_ref = copy(transpose(C_ref))

        # Dimensionality reduction possible
        for (dims, W_test) in [(1,W_svd), (1,W_chol), (2,W_svd), (2,W_chol)]
            is_row = dims == 1

            lda_test = LDM()
            lda_test.Θ.dims = dims
            lda_test.Θ.π = copy(π)
            lda_test.Θ.μ = copy(vec(μ))
            if is_row
                lda_test.Θ.M = copy(transpose(M))
                lda_test.W = copy(transpose(W_test))
            else
                lda_test.Θ.M = copy(M)
                lda_test.W = copy(W_test)
            end

            DA.canonical_coordinates!(lda_test)

            if is_row
                @test isapprox(abs.(lda_test.C), abs.(transpose(C_ref)))
            else
                @test isapprox(abs.(lda_test.C), abs.(C_ref))
            end
        end

        # No dimensionality reduction
        M = random_centroids(T, m, m-1)
        for dims in [1, 2]
            is_row = dims == 1

            lda_test = LDM()
            lda_test.Θ.dims = dims

            lda_test.Θ.M = copy(is_row ? transpose(M) : M)
            lda_test.W = ones(T, m-1, m-1)

            DA.canonical_coordinates!(lda_test)

            @test isapprox(lda_test.C, Matrix{T}(I, m-1, m-1))
        end
    end
end

@testset "fit!(LDA)" begin
    nₘ = [400; 500; 600]
    p = 10
    n = sum(nₘ)
    m = length(nₘ)

    for T in (Float32, Float64)
        X, y, M = random_data(T, nₘ, p)
        π = convert(Vector{T}, nₘ/m)
        Xt = copy(transpose(X))
        Mt = copy(transpose(M))

        π_tests = [nothing, ones(T,m)/m]
        γ_tests = [nothing, range(zero(T), stop=one(T), length=3)...]
        tf_tests = [true, false]

        LDM = DA.LinearDiscriminantModel{T}

        for M_input in [nothing, perturb(M)]
            X_test = copy(X)
            M_test = M_input === nothing ? M : M_input
            Xc = X_test .- M_test[:, y]
            Σ = (Xc*transpose(Xc)) ./ (n-m)

            Xt_test = copy(transpose(X_test))
            Mt_test = copy(transpose(M_test))
            Mt_input = M_input === nothing ? nothing : copy(transpose(M_input))

            for γ in γ_tests 
                if γ !== nothing
                    Σ_test = (1-γ)*Σ + γ*(tr(Σ)/p)*I
                else
                    Σ_test = copy(Σ)
                end
                
                for π_test in π_tests, compute_cov in tf_tests, canonical in tf_tests
                    lda_test = DA.fit!(LDM(), y, copy(Xt_test), dims=1, canonical=canonical, 
                                       compute_covariance=compute_cov, centroids=Mt_input, 
                                       priors=π_test, gamma=γ)
        
                    @test lda_test.Θ.fit == true
                    @test lda_test.Θ.dims == 1
                    @test lda_test.Θ.γ == γ
                    @test isapprox(lda_test.Θ.M, Mt_test)
                    @test isapprox(lda_test.δ₀, det(Σ_test))
                    if compute_cov
                        @test isapprox(lda_test.Θ.Σ, Σ)
                    else
                        @test lda_test.Θ.Σ === nothing
                    end
                    if canonical
                        @test size(lda_test.C) == (p, min(m-1,p))
                    else
                        @test lda_test.C === nothing
                    end

                    lda_test = DA.fit!(LDM(), y, copy(X_test), dims=2, canonical=canonical, 
                                       compute_covariance=compute_cov, centroids=M_input, 
                                       priors=π_test, gamma=γ)
        
                    @test lda_test.Θ.fit == true
                    @test lda_test.Θ.dims == 2
                    @test lda_test.Θ.γ == γ
                    @test isapprox(lda_test.Θ.M, M_test)
                    @test isapprox(lda_test.δ₀, det(Σ_test))
                    if compute_cov
                        @test isapprox(lda_test.Θ.Σ, Σ)
                    else
                        @test lda_test.Θ.Σ === nothing
                    end
                    if canonical
                        @test size(lda_test.C) == (min(m-1,p), p)
                    else
                        @test lda_test.C === nothing
                    end
                end
            end
        end
    end
end