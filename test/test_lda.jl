#@testset "LinearDiscriminantModel" begin
#    n = 20
#    p = 5
#    m = 5
#    for T in (Float32, Float64)
#        LDM = DA.LinearDiscriminantModel{T}
#
#        # test dims argument
#        @test_throws ArgumentError LDM(zeros(T, 3, 3), ones(T, 3)/3, dims=0)
#        @test_throws ArgumentError LDM(zeros(T, 3, 3), ones(T, 3)/3, dims=3)
#
#        # test class priors length/number of classes
#        @test_throws DimensionMismatch LDM(zeros(T, 3, 2), ones(T, 2)/2, dims=1)
#        @test_throws DimensionMismatch LDM(zeros(T, 3, 2), ones(T, 4)/4, dims=1)
#
#        @test_throws DimensionMismatch LDM(zeros(T, 2, 3), ones(T, 2)/2, dims=2)
#        @test_throws DimensionMismatch LDM(zeros(T, 2, 3), ones(T, 4)/4, dims=2)
#
#        # test gamma values
#        lb = zero(T) - eps(zero(T))
#        ub = one(T) + eps(one(T))
#
#        @test_throws DomainError LDM(zeros(T, 3, 1), ones(T, 3)/3, dims=1, gamma=lb)
#        @test_throws DomainError LDM(zeros(T, 3, 1), ones(T, 3)/3, dims=1, gamma=ub)
#
#        # test prior probabilities
#        @test_throws ArgumentError LDM(zeros(T, 3, 1), T[0.3; 0.3; 0.3], dims=1)
#        @test_throws ArgumentError LDM(zeros(T, 3, 1), T[0.4; 0.4; 0.4], dims=1)
#
#        @test_throws DomainError LDM(zeros(T, 3, 1), T[1.0; 0.5; -0.5], dims=1)
#        @test_throws DomainError LDM(zeros(T, 3, 1), T[0.5; 0.5; 0.0], dims=1)
#
#        # Test 1: non-canonical
#        for dims in (1, 2)
#            M = dims == 1 ? ones(T, m, p) : ones(T, p, m)
#            lda_test = LDM(M, ones(T, m)/m, dims=dims, canonical=false)
#
#            @test lda_test.fit == false
#            @test lda_test.dims == dims
#            @test lda_test.W == zeros(T, p, p)
#            @test lda_test.detΣ == zero(T)
#            @test lda_test.M == M
#            @test lda_test.π == ones(T, m)/m
#            @test lda_test.C === nothing
#            @test lda_test.A === nothing
#            @test lda_test.γ === nothing
#        end
#
#        # Test 2: canonical with p > m-1
#        for dims in (1, 2)
#            M = dims == 1 ? ones(T, m, p) : ones(T, p, m)
#            lda_test = LDM(M, ones(T, m)/m, dims=dims, canonical=true, gamma=T(0.5))
#
#            d = min(p, m-1)
#            A_C = dims == 1 ? zeros(T, p, d) : zeros(T, d, p)
#
#            @test lda_test.fit == false
#            @test lda_test.dims == dims
#            @test lda_test.W == zeros(T, p, p)
#            @test lda_test.detΣ == zero(T)
#            @test lda_test.M == M
#            @test lda_test.π == ones(T, m)/m
#            @test lda_test.C == A_C
#            @test lda_test.A == A_C
#            @test lda_test.γ == T(0.5)
#        end
#
#        # Test 3: canonical with p <= m-1
#        for dims in (1, 2)
#            M = dims == 1 ? ones(T, m, m-1) : ones(T, m-1, m)
#            lda_test = LDM(M, ones(T, m)/m, dims=dims, canonical=true, gamma=T(0.5))
#
#            @test lda_test.fit == false
#            @test lda_test.dims == dims
#            @test lda_test.W == zeros(T, m-1, m-1)
#            @test lda_test.detΣ == zero(T)
#            @test lda_test.M == M
#            @test lda_test.π == ones(T, m)/m
#            @test lda_test.C == zeros(T, m-1, m-1)
#            @test lda_test.A == zeros(T, m-1, m-1)
#            @test lda_test.γ == T(0.5)
#        end
#    end
#end


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
            lda_test.dims = 1
            lda_test.M = copy(M)
            lda_test.π = copy(π)
            lda_test.W = copy(W)
            lda_test.μ = copy(vec(μ))

            DA.canonical_coordinates!(lda_test)
            C_res = W*lda_test.C
            @test isapprox(abs.(C_res), abs.(C_ref))
        end

        # Test column-data
        for W in (W_svd, W_chol)
            lda_test = LDM()
            lda_test.dims = 2
            lda_test.M = copy(transpose(M))
            lda_test.π = copy(π)
            lda_test.W = copy(transpose(W))
            lda_test.μ = copy(vec(μ))

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

