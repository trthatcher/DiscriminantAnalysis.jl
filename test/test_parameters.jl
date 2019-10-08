@info "Testing discriminant_parameters.jl"

### Model Parameters

@testset "set_dimensions!(Θ, y, X, dims)" begin
    nₘ = [400; 500; 600]
    n = sum(nₘ)
    m = length(nₘ)
    p = 10

    for T in (Float32, Float64)
        X, y, M = random_data(T, nₘ, p)
        Xt = copy(transpose(X))

        for (dims, X_test) in [(2,X), (1,Xt)]
            Θ_test = DA.DiscriminantParameters{T}()

            Θ_res = DA.set_dimensions!(Θ_test, y, X_test, dims)

            @test Θ_test === Θ_res
            @test Θ_test.dims == dims
            @test Θ_test.m == m
            @test Θ_test.p == p
        end
    end
end

@testset "set_gamma!(Θ, γ)" begin
    for T in (Float32, Float64)
        Θ_test = DA.DiscriminantParameters{T}()

        for γ in range(zero(T), stop=one(T), length=5)
            Θ_res = DA.set_gamma!(Θ_test, γ)

            @test Θ_test === Θ_res
            @test Θ_test.γ == γ
        end

        for γ in [zero(T)-eps(zero(T)), one(T)+eps(one(T))]
            @test_throws DomainError DA.set_gamma!(Θ_test, γ)
        end
    end
end

@testset "set_statistics!(Θ, y, X, centroids)" begin
    nₘ = [400; 500; 600]
    n = sum(nₘ)
    m = length(nₘ)
    p = 10

    for T in (Float32, Float64)
        X, y, M = random_data(T, nₘ, p)
        Xt = copy(transpose(X))
        Mt = copy(transpose(M))

        Θ_test = DA.DiscriminantParameters{T}()
        Θ_test.m = m
        Θ_test.p = p

        Θ_test.dims = 2
        for M_test in [nothing, perturb(M)]
            Θ_res = DA.set_statistics!(Θ_test, y, X, M_test)

            @test Θ_res === Θ_test
            @test isapprox(Θ_test.M, M_test === nothing ? M : M_test)
            @test Θ_test.nₘ == nₘ
        end

        Θ_test.dims = 1
        for Mt_test in [nothing, perturb(Mt)]
            Θ_res = DA.set_statistics!(Θ_test, y, Xt, Mt_test)

            @test Θ_res === Θ_test
            @test isapprox(Θ_test.M, Mt_test === nothing ? Mt : Mt_test)
            @test Θ_test.nₘ == nₘ
        end
    end
end

@testset "set_priors!(Θ, π)" begin
    nₘ = [400; 500; 600]
    n = sum(nₘ)
    m = length(nₘ)

    for T in (Float32, Float64)
        π = convert(Vector{T},nₘ)/n

        Θ_test = DA.DiscriminantParameters{T}()
        Θ_test.m = m
        Θ_test.nₘ = nₘ

        for π_test in [nothing, ones(T,m)/m]
            Θ_res = DA.set_priors!(Θ_test, π_test)

            @test Θ_test === Θ_res
            @test isapprox(Θ_test.π, π_test === nothing ? π : π_test)
        end
    end
end