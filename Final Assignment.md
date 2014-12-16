Coursera Practical Machine Learning Assignment - Benjamin Stauch
========================================================
###Basics
Ok, here we go. We start by loading the packages we need, setting the working directory and importing the data.

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.2
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
setwd("E:/Users/Benjamin/Documents/R/Coursera/repo/Practical-Machine-Learning-Course")

training <- read.csv("pml-training.csv")
testing20 <- read.csv('pml-testing.csv')
```

###Pre-Processing
As we are given a huge number of predictors (160), we start by removing a few. First, we remove those who are irrelevant to the task at hand.

```r
training <- subset(training, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
```
Then, we remove variable which have at least 50% missing data.

```r
training <- training[, colSums(is.na(training)) <= (19622/2)]
```
Next up, we search for variables with near zero variance and remove those as well. This brings us down to 54 variables.

```r
nsv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[,nsv[, 'nzv']==0]
```

###Cross-Validation preparation, building the model and evaluating it
Now we split our data from the training file (the "testing"-file is for the other part of the project) into a training and a testing data set.

```r
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
trainingF <- training[inTrain,]; testingF <- training[-inTrain,]
```
Having done that, let's build the model! Using random forest produces good results in data like the one given to us here. I limit it to 100 trees, as my PC is quite slow.

```r
model <- train(classe ~., method='rf', data=trainingF, ntree=100)
```
We now test the model in the test set and measure its accuracy. 0.9973 is very high, leaving only a measly 0.0027 out of sample error. Success!

```r
pred <- predict(model, testingF)
confusionMatrix(pred, testingF$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    4    0    0    0
##          B    0 1135    0    0    0
##          C    0    0 1026    3    0
##          D    0    0    0  961    2
##          E    0    0    0    0 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9985          
##                  95% CI : (0.9971, 0.9993)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9981          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9965   1.0000   0.9969   0.9982
## Specificity            0.9991   1.0000   0.9994   0.9996   1.0000
## Pos Pred Value         0.9976   1.0000   0.9971   0.9979   1.0000
## Neg Pred Value         1.0000   0.9992   1.0000   0.9994   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1929   0.1743   0.1633   0.1835
## Detection Prevalence   0.2851   0.1929   0.1749   0.1636   0.1835
## Balanced Accuracy      0.9995   0.9982   0.9997   0.9982   0.9991
```
The high accuracy should lead to an error-free prediction in the 20 test samples.

```r
answers <- predict(model,testing20)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



