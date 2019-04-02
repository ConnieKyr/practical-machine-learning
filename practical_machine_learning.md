---
title: "Practical Machine Learning Assignment"
author: "Konstantina Kyriakouli"
date: "April 1, 2019"
output: 
  html_document:
    keep_md: true
---



## Overview:

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. We will explore the data and the measured features, clean it, separate it into test and training set, do some feature selection on the training dataset only, fit various ML models to the training set with 10-fold cross validation repeated ten times, and finally we will assess how well our models perform according to their predictions for the test set.


## Background:

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In order to quantify this question, the participants in the project generating our dataset were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

In more detail, six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

The authors mention that Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. The authors finally made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).

More information can be found here: http://groupware.les.inf.puc-rio.br/har#ixzz5jpuJfgJX

The training data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

**Citation: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.**


## Exploratory Data Analysis and Data cleaning

For the data at hand, we are asked to predict the "classe" variable, using whatever features we deem to be of predictive value. Let's first take a look at our data:


```r
library(dplyr) 
library(stringr)
library(tidyr)
library(naniar)
library(ggplot2)
library(corrplot)
library(wesanderson)
library(knitr)
library(caret)
library(mlbench)
set.seed(7)
tr.data<-read.csv("pml-training.csv")
ts.data<-read.csv("pml-testing.csv")
dim(tr.data)
```

```
## [1] 19622   160
```

```r
dim(ts.data)
```

```
## [1]  20 160
```

```r
#Checking the variable to be predicted:
summary(tr.data$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

We see first, that ~19600 observations are certainly enough for partitioning our data set (we need a separate validation test set partition because the prediction set (test data) given does NOT contain labels). So, the data in the "pml-testing.csv" file will not be used for validation, only for prediction. So, for the time being, we set it completely aside. We need a separate partition of the training data for validation. Furthermore, we see that from the 160 available features not all of them could be useful for predicting classe. There are various techniques we can use for feature reduction/selection. Also we have a lot of missing values, which would have to get removed.

Next, we do some required data cleaning. 
We remove the columns of features that contain only NA values, and also some extra uninformative columns: (user name, time stamps and generally variables not referring to spatial measurements).
All the operations are performed in the training data only, and the prediction test data (the one we need to predict for the quiz) will not get cleaned, but rather, the columns/features we selected from the training data will be selected from the prediction test data before the prediction.


```r
tr.data<-tr.data[, colSums(is.na(tr.data)) == 0]
tr.data<-tr.data[, -(1:7)]
dim(tr.data)
```

```
## [1] 19622    86
```

At this point, we're left with 85 features + 1 class variable.
We noticed that here's still some features that contain empty strings and other characters that have to be interpreted as missing values.
We fix this in the next rows:


```r
na_strings <- c("NA", "", "#DIV/0!")
```


```r
clean_tr <-tr.data %>% replace_with_na_all(condition = ~.x %in% na_strings)
```




```r
#As we see there are now more columns with NA values after these replacements, we repeat:
clean_tr<-clean_tr[, colSums(is.na(clean_tr)) == 0]
#We fix the class labels that were changed to numeric during data cleaning:
clean_tr$classe<-as.factor(clean_tr$classe)
levels(clean_tr$classe)<-c("A", "B", "C", "D", "E")
dim(clean_tr)
```

```
## [1] 19622    53
```

After cleaning the data, we're left with 52 numerical features and 1 class variable.
Looking at the proportions of our class variable, we can see the representation is not equal, as expected. The correct way to do the exercise, Class A, contains more instances than the "faulty" ways. The distribution of the classe variable in the 19622 instances is shown in Appendix Figure 1.

We can further keep only the relevant features for prediction of "classe" with some feature selection and correlation analysis. But before we select features, we need to split our training data into two partitions (70/30), one for feature selection and modeling and one for external validation of the models and obtaining an out-of-sample error rate. It is considered good practice to partition the data set **BEFORE** any sort of feature selection or further manipulation, and keep our downstream processes and our models **BLIND** to the validation/testing partition. Only then we can be certain to avoid overfitting and keep our model reliably **generalizable**.


```r
inTrain<-createDataPartition(clean_tr$classe, p=0.7, list=FALSE)
training<-clean_tr[inTrain,]
testing<-clean_tr[-inTrain,]
#the new parition dimensions are:
dim(training)
```

```
## [1] 13737    53
```

```r
dim(testing)
```

```
## [1] 5885   53
```

The caret function createDataPartition(), is keeping the class representation percentages similar in the partitions as in the initial dataset.

## Feature selection

A pairwise correlation plot of all the 52 features is shown in Appendix Figure 2.
From the correlation plot we see that some features are redundant as they can be highly correlated.
At the same time, it might be useful also to apply a zero variance filter and some univariate and/or multivariate filter.
These conclusions suggest that some basic feature selection might be useful.
We have two ways to perform feature selection: With filters or a wrapper. 
In this case, since we don't have a very restrictively high number of features and simultaneously, we might use models with embedded feature selection, we only used a filter to remove redundant features (correlated with more than 0.9 Pearson correlation coefficient).


```r
out<-findCorrelation(cor(training[,1:52]), cutoff = .9)
training_sel<-training[,-out]
dim(training_sel)
```

```
## [1] 13737    46
```

## ML Model strategy and training

For the model training, we decided to use a random forest model (rf) and a gradient boost machine (gbm) as well as an extreme gradient boosting tree (xgbTree) model from the caret package. Some linear (ensemble) and non-linear (stacking) meta-modelling could be additionally added if needed.
In order to build a robust and reliable model, a scheme of 10-fold cross validation was used. Ideally, we'd use 10 times repeated 10-fold cross-validation. With 19,600 instances and 45 features, this would be computationally expensive though.
 
The models were built with the following code:


```r
#Control parameters:
fitControl<-trainControl(method="cv", number=10, savePredictions = "final", 
                         classProbs = TRUE, verboseIter = TRUE)
#RF model with feature preprocessing:
model_1<-train(x=training_sel[,1:45], y=training_sel$classe, preProcess=c("center", "scale"), method="rf", 
                    trControl = fitControl, tuneLength = 3, metric="ROC")

#gbm model with feature preprocessing:
model_2<-train(x=training_sel[,1:45], y=training_sel$classe, preProcess=c("center", "scale"), method="gbm",
                    trControl = fitControl, tuneLength = 3, metric="ROC")

#xgbTree model with preprocessing:
model_3<-train(x=training_sel[,1:45], y=training_sel$classe, preProcess=c("center", "scale"), method="xgbTree",
                    trControl = fitControl, tuneLength = 3, metric="ROC")

#RF model without preprocessing:
model_4<-train(x=training_sel[,1:45], y=training_sel$classe, method="rf", 
                    trControl = fitControl, tuneLength = 3, metric="ROC")
```

Model out-of-sample error rates during training, derived from the 10-fold CV:




```r
cat("Out of sample error rates during training")
```

```
## Out of sample error rates during training
```

```r
models<-c("RF", "gbm", "xgbTree", "RF_no_preprocessing")
error_rates<-c(1-max(model_1$results['Accuracy']), 1-max(model_2$results['Accuracy']), 1-max(model_3$results['Accuracy']), 1-max(model_4$results['Accuracy']))
kable(data.frame(models, error_rates))
```



models                 error_rates
--------------------  ------------
RF                       0.0073533
gbm                      0.0428048
xgbTree                  0.0064061
RF_no_preprocessing      0.0080072

The models xgbTree and RF with preprocessing give the best accuracy during training.
All the accuracies from the predictions of the hold-out samples at each Fold of the cross-validation process, are averaged and give us in the end the overall Accuracy. This is an estimate of the out-of-sample error rate (1 - Acc).


## ML Model Assessment and Validation

We finally assessed the models on the validation test set:


```r
#preparing the validation set:
testing_sel<-dplyr::select(testing, names(training_sel))
pred_1<-predict(model_1, testing_sel[,1:45])
cf_1<-confusionMatrix(pred_1, testing_sel$classe)
cf_1$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9923534
```

```r
pred_2<-predict(model_2, testing_sel[,1:45])
cf_2<-confusionMatrix(pred_2, testing_sel$classe)
cf_2$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9587086
```

```r
pred_3<-predict(model_3, testing_sel[,1:45])
cf_3<-confusionMatrix(pred_3, testing_sel$classe)
cf_3$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9928632
```

```r
pred_4<-predict(model_4, testing_sel[,1:45])
cf_4<-confusionMatrix(pred_4, testing_sel$classe)
cf_4$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9898046
```

The model RF again seems to give high accuracy in the validation set (good generalization). The version without preprocessing does slightly better for the Classes A, B, and E, while the version with preprocessing predicts slightly better the other two classes (C and D) and has better overall Accuracy. Comparing the Accuracies for the validation test though, we notice that the xgbTree model slightly outperforms the RF one. 
A ranking of variable importance according to the RF model with preprocessing can be seen in Appendix Figure 4.
Also, in Appendix Figure 5 we see a final prediction accuracy on the validation set of the various models.

## Final Conclusions

The models RF with preprocessing (parameters: mtry = 23) and xgbTree with preprocessing (parameters: nrounds = 150, max_depth = 3, eta = 0.4, gamma = 0, colsample_bytree = 0.8, min_child_weight = 1 and subsample = 0.75) show both high accuracies during trainign and better generalization during the validation set prediction. They both gave the full score on the Quiz 4 (so on the predictions for the initial test data with no labels given). If we'd have to choose one, we'd probably choose the RF model, which is parameter-poor (a parameter-rich model is not expected to give low accuracies anyway). 
The prediction results on the unknown labeled data were calculated with the following code:


```r
unknown<-dplyr::select(ts.data, names(training_sel[1:45]))
labels_for_quiz <- predict(model_3, newdata = unknown)
#model_1 and model_3 gave the same labelling
labels_for_quiz
```


## Appendix

**Figure 1: Distribution of classe variable.**


```r
barplot(table(clean_tr$classe))
```

<img src="practical_machine_learning_files/figure-html/unnamed-chunk-14-1.png" width="400px" style="display: block; margin: auto;" />

**Figure 2: Pairwise correlation plot of 52 features.**


```r
corrplot(cor(training[,1:52]), type = "upper", order = "hclust", tl.col = "black", tl.srt = 25)
```

<img src="practical_machine_learning_files/figure-html/unnamed-chunk-15-1.png" height="700px" style="display: block; margin: auto;" />

**Figure 3: RF model training (with preprocessing).**


```r
plot(model_1)
```

<img src="practical_machine_learning_files/figure-html/unnamed-chunk-16-1.png" width="500px" style="display: block; margin: auto;" />

**Figure 4: RF model Variable importance (with preprocessing).**


```r
plot(varImp(model_1))
```

<img src="practical_machine_learning_files/figure-html/unnamed-chunk-17-1.png" width="600px" style="display: block; margin: auto;" />

**Figure 5: ML models Final Accuracies on the validation set.**


```r
cols<-c("RF", "gbm", "xgbTree", "RF_no_preproc")
vals<-c(cf_1$overall[["Accuracy"]], cf_2$overall[["Accuracy"]], cf_3$overall[["Accuracy"]], cf_4$overall[["Accuracy"]])
acc<-data.frame(cols, vals)
colours <- c("firebrick", "dodgerblue", "blue", "yellow")
barplot(acc$vals, main="Overall Accuracies for Validation/Test Set", xlab="models", ylab="Overall Accuracy", names.arg=cols, col=wes_palette("FantasticFox1"))
```

<img src="practical_machine_learning_files/figure-html/unnamed-chunk-18-1.png" width="500px" style="display: block; margin: auto;" />

