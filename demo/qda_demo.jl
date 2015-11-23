data = readcsv(open("iris.data"))  # Read into an array of any

species = unique(data[2:end,5])
to_uid = [species[i] => i for i = 1:length(species)]
to_label = [i => species[i] for i = 1:length(species)]

X = convert(Array{Float64}, data[2:end,1:4])
y = Int64[get(to_uid,species,-1) for species in data[2:end,5]]

using DiscriminantAnalysis

model = qda(X, y)

y_pred = predict(model, X)

y_pred_labels = [get(to_label, uid, "error") for uid in prediction]
