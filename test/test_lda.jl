@testset "LinearDiscriminantModel" begin
    n = 10
    p = 5
    m = 3
    for T in (Float32, Float64)
        LDM = DA.LinearDiscriminantModel{T}

        # test dims argument
        @test_throws ArgumentError LDM(zeros(T, 3, 3), ones(T, 3)/3, dims=0)
        @test_throws ArgumentError LDM(zeros(T, 3, 3), ones(T, 3)/3, dims=3)

        # test class priors length/number of classes
        @test_throws DimensionMismatch LDM(zeros(T, 3, 2), ones(T, 2)/2, dims=1)
        @test_throws DimensionMismatch LDM(zeros(T, 3, 2), ones(T, 4)/4, dims=1)

        @test_throws DimensionMismatch LDM(zeros(T, 2, 3), ones(T, 2)/2, dims=2)
        @test_throws DimensionMismatch LDM(zeros(T, 2, 3), ones(T, 4)/4, dims=2)

        # test gamma values
        lb = zero(T) - eps(zero(T))
        ub = one(T) + eps(one(T))

        @test_throws DomainError LDM(zeros(T, 3, 1), ones(T, 3)/3, dims=1, gamma=lb)
        @test_throws DomainError LDM(zeros(T, 3, 1), ones(T, 3)/3, dims=1, gamma=ub)

        # test prior probabilities
        @test_throws ArgumentError LDM(zeros(T, 3, 1), T[0.3; 0.3; 0.3], dims=1)
        @test_throws ArgumentError LDM(zeros(T, 3, 1), T[0.4; 0.4; 0.4], dims=1)

        @test_throws DomainError LDM(zeros(T, 3, 1), T[1.0; 0.5; -0.5], dims=1)
        @test_throws DomainError LDM(zeros(T, 3, 1), T[1.0; 0.5; -0.5], dims=1)

        @test_throws DomainError LDM(zeros(T, 3, 1), T[0.5; 0.5; 0.0], dims=1)
        @test_throws DomainError LDM(zeros(T, 3, 1), T[0.5; 0.5; 0.0], dims=1)

        # Test 1
        lda_test = LDM(ones(T, m, p), ones(T, m)/m, dims=1, canonical=false)

        @test lda_test.fit == false
        @test lda_test.dims == 1
        @test lda_test.W == zeros(T, p, p)
        @test lda_test.detΣ == zero(T)
        @test lda_test.M == ones(T, m, p)
        @test lda_test.π == ones(T, m)/m
        @test lda_test.C === nothing
        @test lda_test.A === nothing
        @test lda_test.γ === nothing

        # Test 2
        lda_test = LDM(ones(T, p, m), ones(T, m)/m, dims=2, canonical=true, gamma=T(0.5))
        d = min(p, m-1)

        @test lda_test.fit == false
        @test lda_test.dims == 2
        @test lda_test.W == zeros(T, p, p)
        @test lda_test.detΣ == zero(T)
        @test lda_test.M == ones(T, p, m)
        @test lda_test.π == ones(T, m)/m
        @test lda_test.C == zeros(T, d, p)
        @test lda_test.A == zeros(T, d, p)
        @test lda_test.γ == T(0.5)
    end
end


@testset "canonical_coordinates(M, W, dims)" begin
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

        Mc = M .- transpose(π)*M

        Σ_between = Symmetric(transpose(Mc)*(π .* Mc))
        C_ref = eigen(Σ_between, Σ_within).vectors[:, range(p, step=-1, length=m-1)]

        # Test row-data
        for W in (W_svd, W_chol)
            lda_test = LDM(M, π, dims=1, canonical=true)

            copyto!(lda_test.W, W)
            DA.canonical_coordinates!(lda_test)
            C_res = W*lda_test.C
            @test isapprox(abs.(C_res), abs.(C_ref))
        end

        # Test column-data
        for W in (W_svd, W_chol)
            lda_test = LDM(copy(transpose(M)), π, dims=2, canonical=true)

            copyto!(lda_test.W, transpose(W))
            DA.canonical_coordinates!(lda_test)
            C_res = lda_test.C*transpose(W)
            @test isapprox(abs.(C_res), abs.(transpose(C_ref)))
        end

        #copyto!(lda_test.W, W_svd)
        #DA.canonical_coordinates!(lda_test)
        #C_res = W_svd*lda_test.C
        #@test isapprox(abs.(C_res), abs.(C_ref))

        #C_res = W_svd*DA.canonical_coordinates(M, W_svd, π, 1)
        #@test isapprox(abs.(C_res), abs.(C_ref))
        #
        #C_res = W_chol*DA.canonical_coordinates(M, W_chol, π, 1)
        #@test isapprox(abs.(C_res), abs.(C_ref))
        #
        #Mt = transpose(M)
        #W_svd_t = transpose(W_svd)
        #W_chol_t = transpose(W_chol)
        #
        #C_res = DA.canonical_coordinates(Mt, W_svd_t, π, 2)*W_svd_t
        #@test isapprox(transpose(abs.(C_res)), abs.(C_ref))
        #
        #C_res = DA.canonical_coordinates(Mt, W_chol_t, π, 2)*W_chol_t
        #@test isapprox(transpose(abs.(C_res)), abs.(C_ref))
    end
end

# eigen(Σ_between, Σ_within).vectors
#W_svd*transpose(svd((.√(πₖ) .* Mc)*W_svd).Vt)
#W_chol*transpose(svd((.√(πₖ) .* Mc)*W_chol).Vt)

