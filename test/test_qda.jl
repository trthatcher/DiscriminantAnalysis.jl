n_k = [80; 100; 120]
k = length(n_k)
n = sum(n_k)
p = 3

y, X = sampledata(n_k, p)
M = MOD.classmeans(Val{:row}, X, y)
H = MOD.centerclasses!(Val{:row}, copy(X), M, y)
Σ = H'H/(n-1)
Σ_k = 
priors = ones(k)./k

info("Testing ", MOD.classwhiteners!)
for order in (:row, :col)
    for T in FloatingPointTypes
        isrowmajor = order == :row
        H_tst = isrowmajor ? copy(convert(Array{T}, H)) : transpose(convert(Array{T}, H))
        Σ_tst = isrowmajor ? copy(convert(Array{T}, Σ)) : transpose(convert(Array{T}, Σ))
        Σ_k_tst = isrowmajor ? [MOD.covmatrix(Val{:row}, H_tst[y .== i,:]) for i = 1:y.k] :
                               [MOD.covmatrix(Val{:col}, H_tst[:,y .== i]) for i = 1:y.k]

        for γ in (Nullable{T}(), Nullable(convert(T,0.4)), Nullable(one(T)))
            for λ in (Nullable{T}(), Nullable(convert(T,0.4)), Nullable(one(T)))
                W_k = isnull(λ) ? MOD.classwhiteners!(Val{order}, H_tst, y, γ) :
                                  MOD.classwhiteners!(Val{order}, H_tst, y, γ, get(λ))
                for i in eachindex(W_k)
                    S = isnull(λ) ? Σ_k_tst[i] : (1-get(λ))*Σ_k_tst[i] + get(λ)*Σ_tst
                    S = isnull(γ) ? S      : (1-get(γ))*S + get(γ)*trace(S)/p*I
                    if order == :row
                        @test_approx_eq eye(T,p) W_k[i]'*S*W_k[i]
                    else
                        @test_approx_eq eye(T,p) W_k[i]*S*(W_k[i]')
                    end
                end
            end
        end
    end
end

info("Testing ", MOD.qda!)
for order in (:row, :col)
    for T in FloatingPointTypes
        isrowmajor = order == :row
        X_tst = isrowmajor ? copy(convert(Matrix{T}, X)) : transpose(convert(Matrix{T}, X))
        M_tst = isrowmajor ? copy(convert(Matrix{T}, M)) : transpose(convert(Matrix{T}, M))
        H_tst = isrowmajor ? copy(convert(Matrix{T}, H)) : transpose(convert(Matrix{T}, H))

        for γ in (Nullable{T}(), Nullable(convert(T,0.5)))
            for λ in (Nullable{T}(), Nullable(convert(T,0.5)))
                W_k_tst = MOD.qda!(Val{order}, copy(X_tst), copy(M_tst), y, γ, λ)
                W_k_tmp = isnull(λ) ? MOD.classwhiteners!(Val{order}, H_tst, y, γ) :
                                      MOD.classwhiteners!(Val{order}, H_tst, y, γ, get(λ))
                for i in eachindex(W_k_tmp)
                    @test_approx_eq W_k_tst[i] W_k_tmp[i]
                end
            end
        end
    end
end

info("Testing ", MOD.qda)
for order in (:row, :col)
    for T in FloatingPointTypes
        isrowmajor = order == :row
        X_tst = isrowmajor ? copy(convert(Matrix{T}, X)) : transpose(convert(Matrix{T}, X))
        M_tst = isrowmajor ? copy(convert(Matrix{T}, M)) : transpose(convert(Matrix{T}, M))

        for γ in (Nullable{T}(), zero(T), Nullable(convert(T, 0.5)), convert(T, 0.5), one(T))
            for λ in (Nullable{T}(), zero(T), Nullable(convert(T, 0.5)), convert(T, 0.5), one(T))
                model = MOD.qda(X_tst, y, order=Val{order}, gamma=γ, lambda=λ)
                γ_tst = isa(γ, Nullable) ? γ : Nullable(γ)
                λ_tst = isa(λ, Nullable) ? λ : Nullable(λ)
                W_k_tst = MOD.qda!(Val{order}, copy(X_tst), copy(M_tst), y, γ_tst, λ_tst)
                for i in eachindex(model.W_k)
                    @test_approx_eq model.W_k[i] W_k_tst[i]
                end
                @test_approx_eq model.M M_tst
            end
        end
    end
end


#=

info("Testing ", MOD.discriminants_qda)
for T in FloatingPointTypes
    X_tmp = copy(convert(Matrix{T}, X))
    priors_tmp = convert(Vector{T}, priors)

    model = qda(X_tmp, y)
    Z2 = hcat([MOD.dotrows((X_tmp .- M[i,:])*model.W_k[i]) for i in eachindex(priors_tmp)]...)
    δ = -Z2/2 .+ log(priors_tmp)'
    @test_approx_eq δ MOD.discriminants(model, X_tmp)

    for γ in (zero(T), convert(T, 0.25), convert(T, 0.75), one(T))
        model = qda(X_tmp, y)
        Z2 = hcat([MOD.dotrows((X_tmp .- M[i,:])*model.W_k[i]) for i in eachindex(priors_tmp)]...)
        δ = -Z2/2 .+ log(priors_tmp)'
        @test_approx_eq δ MOD.discriminants(model, X_tmp)
    end
end

info("Testing ", MOD.ModelQDA)
show(DevNull, qda(X, y))
=#
