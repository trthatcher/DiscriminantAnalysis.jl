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

        LDM = DA.LinearDiscriminantModel{T}

        for (dims, M_input) in [(1,nothing), (1,perturb(Mt)), (2,nothing), (2,perturb(M))]
            if dims == 1
                X_test = copy(Xt)
                M_test = M_input === nothing ? copy(Mt) : M_input

                Xc = X_test .- M_test[y, :]
                Σ = (transpose(Xc)*Xc) ./ (n-m)
            else
                X_test = copy(X)
                M_test = M_input === nothing ? copy(M) : M_input

                Xc = X_test .- M_test[:, y]
                Σ = (Xc*transpose(Xc)) ./ (n-m)
            end

            for π_test in π_tests, γ_test in γ_tests
                if !(γ_test === nothing)
                    Σ_test = (1-γ_test)*Σ + γ_test*(tr(Σ)/p)*I
                else
                    Σ_test = copy(Σ)
                end

                lda_test = DA.fit!(LDM(), y, copy(X_test), dims=dims, canonical=false, 
                                   compute_covariance=true, centroids=M_input, 
                                   priors=π_test, gamma=γ_test)
    
                @test lda_test.Θ.fit == true
                @test lda_test.Θ.dims == dims
                @test lda_test.Θ.γ == γ_test
                @test isapprox(lda_test.Θ.M, M_test)
                @test isapprox(lda_test.Θ.Σ, Σ)
                @test isapprox(lda_test.δ, det(Σ_test))
            end
        end
    end
end
# eigen(Σ_between, Σ_within).vectors
#W_svd*transpose(svd((.√(πₖ) .* Mc)*W_svd).Vt)
#W_chol*transpose(svd((.√(πₖ) .* Mc)*W_chol).Vt)