#~%~%~%~%~%~%  Da Response types %~%~%~%~%~%~% 

abstract ModResp

type DaResp{T<:FP} <: ModResp
	y::PooledDataArray	# Response vector
	priors::Vector{Float64}	# Prior weights
	counts::Vector{Int64}	# Prior observation counts
	function DaResp(y::PooledDataArray, priors::Vector{T})
		k = length(priors)
		k == length(levels(y)) || error("Length mismatch priors/levels")
		n = length(y)
		c = zeros(Int64,k)
		for i = 1:n
			c[y.refs[i]] += 1
		end
		new(y, priors, c)
	end
end

#~%~%~%~%~%~% Discriminant function types %~%~%~%~%~%

abstract Discr

type RegDiscr <: Discr
	whiten::Array{Float64,3}
	lambda::Float64
	gamma::Real
end

type LinDiscr <: Discr
	whiten::Matrix{Float64}
	gamma::Real
end

type QuadDiscr <: Discr
	whiten::Array{Float64,3}
	gamma::Real
end

#~%~%~%~%~%~%  Da predictor types %~%~%~%~%~%~% 

abstract DaPred

type RdaPred{T<:Discr} <: DaPred
	X::Matrix{Float64}
	means::Matrix{Float64}
	discr::T
	logpr::Vector{Float64}
end

#~%~%~%~%~%~% Da model types %~%~%~%~%~%~% 

type DaModel
	mf::ModelFrame
	dr::DaResp
	dp::DaPred
	f::Formula
end
