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
    end
end

@testset "_fit!(LDA)" begin
    nₘ = [40; 50; 60]
    p = 5
    n = sum(nₘ)
    m = length(nₘ)

    for T in (Float32, Float64)
        X, y, M = random_data(T, nₘ, p)
        π = convert(Vector{T}, nₘ/m)
        scale = range(convert(T, 0.95), step=convert(T, 0.001), stop=convert(T, 1.1))

        M_test_list = [nothing, rand(scale, m, p) .* M]
        π_test_list = [nothing, ones(T,m)/m]
        γ_test_list = [nothing, zero(T), convert(T, 0.5), one(T)]

        LDM = DA.LinearDiscriminantModel{T}

        for M_test in M_test_list, π_test in π_test_list, γ_test in γ_test_list
            Xc = X .- (M_test === nothing ? M[y, :] : M_test[y, :])
            Σ = (transpose(Xc)*Xc) ./ (n-m)

            if !(γ_test === nothing)
                Σ = (1-γ_test)*Σ + γ_test*(tr(Σ)/p)*I
            end

            lda_test = DA._fit!(LDM(), y, copy(X), 1, false, M_test, π_test, γ_test)

            @test lda_test.fit == true
            @test lda_test.dims == 1
            @test isapprox(lda_test.M, M_test === nothing ? M : M_test)
            @test isapprox(lda_test.detΣ, det(Σ))
        end
    end
end
# eigen(Σ_between, Σ_within).vectors
#W_svd*transpose(svd((.√(πₖ) .* Mc)*W_svd).Vt)
#W_chol*transpose(svd((.√(πₖ) .* Mc)*W_chol).Vt)