### Name: Ayelawa Samuel

### Title of dataset used was "Cardiovascualar disease" Using R.


### Task to perform
# 1. Do general descriptive analysis

# 2. Do predictive analysis on the data. (Logistic Regression)


#read the dataset in R

CVD<-read.csv(file.choose(), header=T)

### I am using a cardiovascular dataset containing 70000 observations and 13 variables as describe below

# Participants Id

# Age in days

# Gender

# height

# weight

# ap_hi (systolic blood pressure)

# ap_lo (diastolic blood pressure)

# cholesterol

# gluc (glucose)

# smoke

# alco (alcohol)

# active (physical activity)

# cardio (cardiovascular disease)



#loading the library to be used in the project

library(dplyr)

library (caTools)

#check data

str(CVD)

#dataset dimension

dim(CVD)

# missing value
sum(is.na(CVD))

# some data manipulation so as to fit into my algorithm

## Remove unwanted variable

CVD<-CVD%>% select(-1)

### change the age in days to years using

Age<-(CVD$age/365)

### round up the decimal point into single number

CVD$Age<-round(as.numeric(CVD$Age), )

#conversion of intergers to factors

CVD$gender<-factor(CVD$gender, labels=c("Female", "male"))

CVD$cholesterol<-factor(CVD$cholesterol, labels = c("Normal", "Above normal", "Well above normal"))

CVD$gluc<-factor(CVD$gluc, labels = c("Normal", "Above normal", "Well above normal"))

CVD$smoke<-factor(CVD$smoke, labels = c("No", "Yes"))

CVD$alco<-factor(CVD$alco, labels = c("No", "Yes"))

CVD$active<-factor(CVD$active, labels = c("No", "Yes"))

CVD$cardio<-factor(CVD$cardio, labels = c("No", "Yes"))


## Division of dataset in Train_dataset and Test_dataset

set.seed(100)

library(caTools)

sample<-sample.split(CVD$cardio, SplitRatio = 0.65)

Train_dataset<-subset(CVD, sample==T)

Test_dataset<-subset(CVD, sample==F)



## Logistic Regression

model<-glm(cardio~. ,data = Train_dataset, family = "binomial")

## Baseline Accuracy of Train_dataset

prop.table(table(Train_dataset$cardio))

## is 49.97 = 50%

pred_test<-predict(model, Test_dataset, type="response")


# Confusion matrix of the algorithm

final<-table(Test_dataset$cardio, pred_test>=0.05)

## Accuracy of the Test_dataset

(41+12203)/(12216+40)

# 99.9%.

#In conclusion

# The baseline accuracy for the data was 49.97%. while the accuracy on the test data was 99.9%. Overall, the logistic regression is beating
 
 the baseline accuracy by a big margin on the test datasets, and the results are good.