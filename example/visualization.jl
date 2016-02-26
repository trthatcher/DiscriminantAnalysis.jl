using DiscriminantAnalysis, Gadfly

### Helper functions ###

function rotationmatrix2D{T<:AbstractFloat}(θ::T)
    T[cos(θ) -sin(θ);
      sin(θ)  cos(θ)]
end

function boxmuller(n::Integer)  # Generates two normally distributed variables
    u1 = rand(n)
    u2 = rand(n)
    Z = Float64[(√(-2log(u1)) .* cos(2π*u2)) (√(-2log(u1)) .* sin(2π*u2))]
end

function boundary(model, xrange, yrange)  # Create the decision boundary 
    Z = hcat(vec(Float64[x for x in xrange, y in yrange]), 
             vec(Float64[y for x in xrange, y in yrange]))
    δ = DiscriminantAnalysis.discriminants(model, Z)
    Z = reshape(δ[:,1] - δ[:,2], length(xrange), length(yrange))
    Contour.coordinates(Contour.contours(xrange, yrange, Z, 0.0)[1].lines[1])
end


### Sample Data ###

n = 250

Z1 = boxmuller(n)
σ1 = [0.5 2.0]
X1 = ((Z1 .* σ1) .- [0.0 4.25]) * rotationmatrix2D(π/4) 

Z2 = boxmuller(n)
σ2 = [3.0 1.5]
X2 = ((Z2 .* σ2) .+ [0.0 2.25]) * rotationmatrix2D(π/4)

X = vcat(X1,X2)
y = repeat([1,2], inner=[n])

xmin = minimum(X[:,1])
xmax = maximum(X[:,1])

ymin = minimum(X[:,2])
ymax = maximum(X[:,2])

aspect = (ymax-ymin)/(xmax-xmin)

m = 150  # Used for interpolating the decision boundary
xrange = linspace(xmin,xmax,m)
yrange = linspace(ymin,ymax,m)


### Plots ###

for (obj, desc) in ((:lda, "Linear Discriminant Analysis"), 
                    (:qda, "Quadratic Discriminant Analyisis"))
    @eval begin
        model = ($obj)(X, y)
        cx, cy = boundary(model, xrange, yrange)

        P = plot(
                x = vec(X[:,1]), 
                y = vec(X[:,2]),
                color = map(class -> "Class $class", y), 
                Geom.point,
                Scale.color_discrete_manual(colorant"red",colorant"blue"),
                Guide.XLabel("X Variable"),
                Guide.YLabel("Y Variable"),
                Guide.title($desc),
                Guide.colorkey(""),
                Coord.Cartesian(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
            )
        L = layer(x=cx, y=cy, Geom.line(preserve_order=true),
                  Theme(default_color=colorant"black", line_width=.4mm))
        unshift!(P.layers,L[1])

        draw(PNG($(string(obj)) * ".png", 6inch, (6*aspect)inch), P)
    end
end
