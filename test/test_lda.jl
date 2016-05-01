n_k = [80; 100; 120]
k = length(n_k)
n = sum(n_k)
p = 4  # make greater than or equal to k

y, X = sampledata(n_k, p)
M = MOD.classmeans(Val{:row}, X, y)
H = MOD.centerclasses!(Val{:row}, copy(X), M, y)
Σ = H'H/(n-1)
priors = ones(k)./k

info("Testing ", MOD.lda!)
for order in (:row, :col)
    for T in FloatingPointTypes
        isrowmajor = order == :row
        X_tst = isrowmajor ? copy(convert(Matrix{T}, X)) : transpose(convert(Matrix{T}, X))
        M_tst = isrowmajor ? copy(convert(Matrix{T}, M)) : transpose(convert(Matrix{T}, M))
        H_tst = isrowmajor ? copy(convert(Matrix{T}, H)) : transpose(convert(Matrix{T}, H))
        Σ_tst = isrowmajor ? copy(convert(Matrix{T}, Σ)) : transpose(convert(Matrix{T}, Σ))

        for γ in (Nullable{T}(), Nullable(zero(T)), Nullable(convert(T,0.5)), Nullable(one(T)))
            W_tmp = MOD.lda!(Val{order}, copy(X_tst), M_tst, y, γ)
            S = isnull(γ) ? Σ_tst : (1-get(γ))*Σ_tst + (get(γ)/p)*trace(Σ_tst)*I
            if isrowmajor
                @test_approx_eq (W_tmp')*S*W_tmp eye(T,p)  # Validate whitening of covariance
            else
                @test_approx_eq W_tmp*S*(W_tmp') eye(T,p)
            end
        end
        γ = Nullable{T}()

        # Check that dimension problems are caught
        X_tst1 = isrowmajor ? rand(T,1,p) : rand(T,p,1)
        X_tst2 = isrowmajor ? rand(T,n,0) : rand(T,0,n)
        @test_throws DimensionMismatch MOD.lda!(Val{order}, X_tst1, M_tst, y, γ)
        @test_throws DimensionMismatch MOD.lda!(Val{order}, X_tst2, M_tst, y, γ)
    end
end

info("Testing ", MOD.cda!)
for order in (:row, :col)
    for T in FloatingPointTypes
        isrowmajor = order == :row
        X_tst = isrowmajor ? copy(convert(Matrix{T}, X)) : transpose(convert(Matrix{T}, X))
        M_tst = isrowmajor ? copy(convert(Matrix{T}, M)) : transpose(convert(Matrix{T}, M))
        H_tst = isrowmajor ? copy(convert(Matrix{T}, H)) : transpose(convert(Matrix{T}, H))
        Σ_tst = isrowmajor ? copy(convert(Matrix{T}, Σ)) : transpose(convert(Matrix{T}, Σ))
        priors_tst = copy(convert(Vector{T}, priors)) 

        # Regular model tests -> Validated that covariance matrices are whitened
        for γ in (Nullable{T}(), Nullable(zero(T)), Nullable(convert(T,0.5)), Nullable(one(T)))
            S = isnull(γ) ? Σ_tst : (1-get(γ))*Σ_tst + (get(γ)/p)*trace(Σ_tst)*I
            W_tmp = MOD.cda!(Val{order}, copy(X_tst), copy(M_tst), y, γ, priors_tst)
            if isrowmajor
                @test_approx_eq (W_tmp')*S*W_tmp eye(T,k-1)
            else
                @test_approx_eq W_tmp*S*(W_tmp') eye(T,k-1)
            end
        end
        γ = Nullable{T}()

        # Check that LDA solution returned when p < k
        X_tst = isrowmajor ? copy(convert(Matrix{T}, X[:,1:k-1])) :
                             transpose(convert(Matrix{T}, X[:,1:k-1]))
        M_tst = isrowmajor ? copy(convert(Matrix{T}, M[:,1:k-1])) :
                             transpose(convert(Matrix{T}, M[:,1:k-1]))
        W_tmp = MOD.cda!(Val{order}, copy(X_tst), copy(M_tst), y, γ, priors_tst)
        W_tst = MOD.lda!(Val{order}, copy(X_tst), copy(M_tst), y, γ)
        @test_approx_eq W_tmp W_tst

        # Check that dimension problems are caught
        X_tst1 = isrowmajor ? rand(T,1,p) : rand(T,p,1)
        X_tst2 = isrowmajor ? rand(T,n,0) : rand(T,0,n)
        @test_throws DimensionMismatch MOD.cda!(Val{order}, X_tst1, M_tst, y, γ, priors_tst)
        @test_throws DimensionMismatch MOD.cda!(Val{order}, X_tst2, M_tst, y, γ, priors_tst)
        @test_throws DimensionMismatch MOD.cda!(Val{order}, X_tst,  M_tst, y, γ, rand(T,k+1))
    end
end

info("Testing ", MOD.lda)
for order in (:row, :col)
    for T in FloatingPointTypes
        isrowmajor = order == :row
        X_tst = isrowmajor ? copy(convert(Matrix{T}, X)) : transpose(convert(Matrix{T}, X))
        M_tst = isrowmajor ? copy(convert(Matrix{T}, M)) : transpose(convert(Matrix{T}, M))

        for γ in (Nullable{T}(), zero(T), Nullable(convert(T, 0.5)), convert(T, 0.5), one(T))
            model = MOD.lda(X_tst, y, order=Val{order}, gamma=γ)
            γ_tst = isa(γ, Nullable) ? γ : Nullable(γ)
            W_tst = MOD.lda!(Val{order}, copy(X_tst), copy(M_tst), y, γ_tst)
            @test model.is_cda == false
            @test_approx_eq model.W W_tst
            @test_approx_eq model.M M_tst
        end
    end
end

info("Testing ", MOD.cda)
for order in (:row, :col)
    for T in FloatingPointTypes
        isrowmajor = order == :row
        X_tst = isrowmajor ? copy(convert(Matrix{T}, X)) : transpose(convert(Matrix{T}, X))
        M_tst = isrowmajor ? copy(convert(Matrix{T}, M)) : transpose(convert(Matrix{T}, M))
        priors_tst = copy(convert(Vector{T}, priors))

        for γ in (Nullable{T}(), zero(T), Nullable(convert(T, 0.5)), convert(T, 0.5), one(T))
            model = MOD.cda(X_tst, y, order=Val{order}, gamma=γ)
            γ_tst = isa(γ, Nullable) ? γ : Nullable(γ)
            W_tst = MOD.cda!(Val{order}, copy(X_tst), copy(M_tst), y, γ_tst, priors_tst)
            @test model.is_cda == true
            @test_approx_eq model.W W_tst
            @test_approx_eq model.M M_tst
        end
    end
end

info("Testing ", MOD.discriminants_lda)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    priors_tmp = convert(Vector{T}, priors)

    model1 = lda(X_tmp,  y, order=Val{:row})
    model2 = lda(X_tmp', y, order=Val{:col})
    D = hcat(-[MOD.dotvectors(Val{:row}, (X_tmp .- M[i,:])*model1.W)/2
                for i in eachindex(priors_tmp)]...) .+ log(priors_tmp)'

    @test_approx_eq D  MOD.discriminants(model1, X_tmp)
    @test_approx_eq D' MOD.discriminants(model2, X_tmp')
end

info("Testing ", MOD.classify)
for order in (:row, :col)
    for T in FloatingPointTypes
        isrowmajor = order == :row
        X_tst = isrowmajor ? copy(convert(Matrix{T}, X)) : transpose(convert(Matrix{T}, X))
        priors_tst = convert(Vector{T}, priors)

        model1 = lda(X_tst, y, order=Val{order})
        model2 = cda(X_tst, y, order=Val{order})

        D1 = vec(mapslices(indmax, MOD.discriminants(model1, X_tst), isrowmajor ? 2 : 1))
        D2 = vec(mapslices(indmax, MOD.discriminants(model2, X_tst), isrowmajor ? 2 : 1))

        @test D1 == MOD.classify(model1, X_tst)
        @test D2 == MOD.classify(model2, X_tst)
    end
end

info("Testing ", MOD.ModelLDA)
show(DevNull, lda(X, y, order=Val{:row}))
