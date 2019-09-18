@info "Testing lda.jl"

@testset "canonical_coordinates!(LDA)" begin
    p = 10
    m = 5

    for T in (Float32, Float64)
        LDM = DA.LinearDiscriminantModel{T}

        Σ_within, W_svd, W_chol = random_cov(T, p)

        π = rand(T, m)
        π ./= sum(π)

        M = zeros(T, m, p)
        for k = 1:m
            M[k, :] = transpose(T[rand() < 0.5 ? -2k : 2k for i = 1:p])
        end
        μ = transpose(π)*M

        Mc = M .- μ

        Σ_between = Symmetric(transpose(Mc)*(π .* Mc))
        C_ref = eigen(Σ_between, Σ_within).vectors[:, range(p, step=-1, length=m-1)]

        # Test row-data
        for W in (W_svd, W_chol)
            lda_test = LDM()
            lda_test.Θ.dims = 1
            lda_test.Θ.M = copy(M)
            lda_test.Θ.π = copy(π)
            lda_test.Θ.μ = copy(vec(μ))
            lda_test.W = copy(W)

            DA.canonical_coordinates!(lda_test)
            C_res = W*lda_test.C
            @test isapprox(abs.(C_res), abs.(C_ref))
        end

        # Test column-data
        for W in (W_svd, W_chol)
            lda_test = LDM()
            lda_test.Θ.dims = 2
            lda_test.Θ.M = copy(transpose(M))
            lda_test.Θ.π = copy(π)
            lda_test.Θ.μ = copy(vec(μ))
            lda_test.W = copy(transpose(W))

            DA.canonical_coordinates!(lda_test)
            C_res = lda_test.C*transpose(W)
            @test isapprox(abs.(C_res), abs.(transpose(C_ref)))
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
        scale = range(convert(T, 0.95), step=convert(T, 0.001), stop=convert(T, 1.1))

        M_test_list = [nothing, rand(scale, p, m) .* M]
        π_test_list = [nothing, ones(T,m)/m]
        γ_test_list = [nothing, zero(T), convert(T, 0.5), one(T)]

        LDM = DA.LinearDiscriminantModel{T}

        for M_test in M_test_list, π_test in π_test_list, γ_test in γ_test_list
            Xc = X .- (M_test === nothing ? M[:, y] : M_test[:, y])
            Σ = (Xc*transpose(Xc)) ./ (n-m)

            if !(γ_test === nothing)
                Σ = (1-γ_test)*Σ + γ_test*(tr(Σ)/p)*I
            end

            lda_test = DA.fit!(LDM(), y, copy(X), 2, false, M_test, π_test, γ_test)

            @test lda_test.Θ.fit == true
            @test lda_test.Θ.dims == 2
            @test isapprox(lda_test.Θ.M, M_test === nothing ? M : M_test)
            @test isapprox(lda_test.Θ.detΣ, det(Σ))
        end
    end
end
# eigen(Σ_between, Σ_within).vectors
#W_svd*transpose(svd((.√(πₖ) .* Mc)*W_svd).Vt)
#W_chol*transpose(svd((.√(πₖ) .* Mc)*W_chol).Vt)