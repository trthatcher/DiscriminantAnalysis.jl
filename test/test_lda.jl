@testset "canonical_coordinates(M, W, dims)" begin
    p = 3
    k = 3

    for T in (Float32, Float64)
        Σ_within, W_svd, W_chol = random_cov(T, p)

        M = rand(T, k, p) .+ Vector{T}(0:2:2k-2)
        π = rand(T, p)
        π ./= sum(π)
        M  .-= transpose(π)*M

        Σ_between = Symmetric(transpose(M)*(π .* M))

        C_ref = eigen(Σ_between, Σ_within).vectors[:, k:-1:2]

        C_res = W_svd*DA.canonical_coordinates(M, W_svd, π, 1)
        @test isapprox(abs.(C_res), abs.(C_ref))

        C_res = W_chol*DA.canonical_coordinates(M, W_chol, π, 1)
        @test isapprox(abs.(C_res), abs.(C_ref))

        Mt = transpose(M)
        W_svd_t = transpose(W_svd)
        W_chol_t = transpose(W_chol)

        C_res = DA.canonical_coordinates(Mt, W_svd_t, π, 2)*W_svd_t
        @test isapprox(transpose(abs.(C_res)), abs.(C_ref))

        C_res = DA.canonical_coordinates(Mt, W_chol_t, π, 2)*W_chol_t
        @test isapprox(transpose(abs.(C_res)), abs.(C_ref))
    end
end

# eigen(Σ_between, Σ_within).vectors
#W_svd*transpose(svd((.√(πₖ) .* Mc)*W_svd).Vt)
#W_chol*transpose(svd((.√(πₖ) .* Mc)*W_chol).Vt)

