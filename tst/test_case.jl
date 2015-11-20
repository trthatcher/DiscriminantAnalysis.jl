using DiscriminantAnalysis;
nc = 4; 
p = 3; 
k = 4; 
y = repeat(Int64[i for i = 1:k],inner=[nc],outer=[1]);  
X = vcat([rand(nc,p) .+ 10rand(1,3) for i = 1:k]...);
mod = qda(X,y)
