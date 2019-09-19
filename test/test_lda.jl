@info "Testing lda.jl"

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
                #C_res = transpose(W_test)*lda_test.C
                #@test isapprox(abs.(C_res), abs.(transpose(C_ref)))
            else
                @test isapprox(abs.(lda_test.C), abs.(C_ref))
                #C_res = lda_test.C*W_test
                #@test isapprox(abs.(C_res), abs.(C_ref))
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
            #@test isapprox(lda_test.A, ones(T, m-1, m-1))
        end
    end
end

#@testset "fit!(LDA)" begin
#    nₘ = [400; 500; 600]
#    p = 10
#    n = sum(nₘ)
#    m = length(nₘ)
#
#    for T in (Float32, Float64)
#        X, y, M = random_data(T, nₘ, p)
#        π = convert(Vector{T}, nₘ/m)
#        scale = range(convert(T, 0.95), step=convert(T, 0.001), stop=convert(T, 1.1))
#
#        M_test_list = [nothing, rand(scale, p, m) .* M]
#        π_test_list = [nothing, ones(T,m)/m]
#        γ_test_list = [nothing, zero(T), convert(T, 0.5), one(T)]
#
#        LDM = DA.LinearDiscriminantModel{T}
#
#        for M_test in M_test_list, π_test in π_test_list, γ_test in γ_test_list
#            Xc = X .- (M_test === nothing ? M[:, y] : M_test[:, y])
#            Σ = (Xc*transpose(Xc)) ./ (n-m)
#
#            if !(γ_test === nothing)
#                Σ = (1-γ_test)*Σ + γ_test*(tr(Σ)/p)*I
#            end
#
#            lda_test = DA.fit!(LDM(), y, copy(X), 2, false, M_test, π_test, γ_test)
#
#            @test lda_test.Θ.fit == true
#            @test lda_test.Θ.dims == 2
#            @test isapprox(lda_test.Θ.M, M_test === nothing ? M : M_test)
#            @test isapprox(lda_test.Θ.detΣ, det(Σ))
#        end
#    end
#end
# eigen(Σ_between, Σ_within).vectors
#W_svd*transpose(svd((.√(πₖ) .* Mc)*W_svd).Vt)
#W_chol*transpose(svd((.√(πₖ) .* Mc)*W_chol).Vt)