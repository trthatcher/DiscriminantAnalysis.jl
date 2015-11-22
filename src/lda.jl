# %~%~%~%~%~%~%~%~%~% Helper Functions %~%~%~%~%~%~%~%~%

# Perform linear discriminant analysis (rank-reduced is default)
#=
function fitda!(dr::DaResp, dp::RdaPred{LinDiscr}; tol::Float64=0.0001)
	nk = length(dr.counts)
	n, p = size(dp.X)
	Xc, sd = centerscalematrix(dp.X,dp.means,dr.y.refs)
	s, V = svd(Xc,false)[2:3]
	if length(s) < p s =  vcat(s, zeros(p - length(s))) end
	if dp.discr.gamma != 0 	# Shrink towards (I * Average Eigenvalue)
		s = (s .^ 2)/(n-nk) .* (1-dp.discr.gamma) .+ (dp.discr.gamma * sum(s) / p)
	else	# No shrinkage
		s = (s .^ 2)/(n-nk)
	end
	rank = sum(s .> s[1]*tol)
	rank == p || error("Rank deficiency detected with tolerance=$tol.")
	dp.discr.whiten = diagm(1 ./ sd) * V * diagm(1 ./ sqrt(s))
	if (dp.discr.rrlda == true) & (nk > 2)
		mu = sum(dr.priors .* dp.means, 1)
		Mc = (dp.means .- mu) * dp.discr.whiten
		s, V = svd(Mc, false)[2:3]
		rank = sum(s .> s[1]*tol)
		dp.discr.whiten = dp.discr.whiten * V[:,1:rank]
	end
end
=#


# %~%~%~%~%~%~%~%~%~% Prediction Methods %~%~%~%~%~%~%~%~%
#=
function predict(mod::DaModel, X::Matrix{Float64})
	D = index_to_level(mod.dr.y)
	return PooledDataArray(map(x->get(D,convert(Uint32,x),0), pred(mod.dp, X)))
end

function predict(mod::DaModel, df::AbstractDataFrame)
	X = ModelMatrix(ModelFrame(mod.f, df)).m[:,2:]
	return predict(mod, X)
end

function pred(dp::RdaPred{LinDiscr}, X::Matrix{Float64})
	n,p = size(X)
	ng = length(dp.logpr)
	Zk = Array(Float64,n,p)
	P = Array(Float64, n, ng)
	for k=1:ng
		Zk = (X .- dp.means[k,:]) * dp.discr.whiten
		P[:,k] = mapslices(x -> -0.5*sum(x .^ 2), Zk, 2) .+ dp.logpr[k]
	end
	return mapslices(indmax,P,2)
end
=#
