using DataFrames

data = readcsv("iris.csv")

predictor = data[:,1:4]

classes =  data[:,5] # map(x -> convert(Integer x), data[:,5])

classes = map(x -> convert(Integer, x), classes)

classlist = unique(classes)

k = length(classlist)

p, n = size(predictor)

#Pooled group is group 0

cov_mat = Array(Float64, p, p, k + 1)

mean_mat = Array(Float64, p, k)

for i in (1:k)

end


