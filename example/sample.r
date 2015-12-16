library(MASS)
df <- read.table("iris.csv", header = TRUE, sep=",")

fit <- lda(Species ~ PetalWidth + SepalWidth, data=df)

sum(predict(fit, newdata=df)$class == df$Species)
