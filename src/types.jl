#~%~%~%~%~%~%  Da Response types %~%~%~%~%~%~% 

abstract ModResp

type DaResp <: ModResp
	y::PooledDataArray	# Response vector
	priors::Vector{Float64}	# Prior weights
	counts::Vector{Int64}	# Prior observation counts
	function DaResp(y::PooledDataArray, priors::Vector{Float64})
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
	rrlda::Bool
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

#~%~%~%~%~%~% IO %~%~%~%~%~%~% 

function printdescribe(dr::DaResp)
	println("Response:\n")
	println(DataFrame(hcat(levels(dr.y), dr.priors, dr.counts), ["Group", "Prior","Count"]))
end

function printdescribe(d::RegDiscr)
	println("Lambda: $(d.lambda)")
	println("Gamma: $(d.gamma)")
end

printdescribe(d::QuadDiscr) = println("Gamma: $(d.gamma)")

function printdescribe(d::LinDiscr)
	println("Lambda: $(d.lambda)")
	println("Rank-reduced: $(d.rrlda)")
end

function Base.show(io::IO, mod::DaModel)
	println(mod.f)
	print("\n")
	printdescribe(mod.dr)
	print("\n")
	printdescribe(mod.dp.discr)
	println("Group means:")
	println(DataFrame(hcat(levels(mod.dr.y),mod.dp.means), vcat("Group", coefnames(mod.mf)[2:])))
end
