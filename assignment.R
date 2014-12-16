library(caret)
library(randomForest)
setwd("E:/Users/Benjamin/Documents/R/Coursera/repo/Practical-Machine-Learning-Course")

training <- read.csv("pml-training.csv")
testing20 <- read.csv('pml-testing.csv')

training <- subset(training, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
training <- training[, colSums(is.na(training)) <= (19622/2)]
nsv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[,nsv[, 'nzv']==0]

inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
trainingF <- training[inTrain,]; testingF <- training[-inTrain,]

model <- train(classe ~., method='rf', data=trainingF, ntree=100)
pred <- predict(model, testingF)
confusionMatrix(pred, testingF$classe)
answers <- predict(model,testing20)
answers

