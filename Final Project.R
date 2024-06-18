library(fastDummies)
library(caret)
library(corrplot)
library(e1071)
library(pls)
library(klaR)
library(kernlab)
library(ggplot2)
library(doParallel)

ncores<-4
cl<-makePSOCKcluster(ncores)
registerDoParallel(cl)

pre_data <- read.csv("C:/Users/vinay/OneDrive/Desktop/PredectiveModeling/bodyPerformance.csv")

#Checking for missing values
sapply(pre_data,function(x) sum(is.na(x)))
Response<-as.factor(pre_data$class)

#check BMI column
pre_data$BMI<- round(pre_data$weight_kg / ((pre_data$height_cm/100)^2),2)
pre_data

#dummy variables
pre_data <- dummy_cols(pre_data,select_columns ="gender")
pre_data<-pre_data[,-which(names(pre_data)=="gender")]
pre_data <- pre_data[, c(1,2,3,4,5,6,7,8,9,10,12,11,13,14)]
pre_data

#Gender distribution
male<-round(sum(pre_data$gender_M)/nrow(pre_data),2)*100
female<-round(sum(pre_data$gender_F)/nrow(pre_data),2)*100
gender<-c(male,female)
gender

par(mfrow= c(1,1))
gender_dist<-barplot(table(pre_data$gender_F), main="Gender Distribution",xlab =
                       "Gender(M=0, F=1)",ylab="Count")

# Add labels with percentages inside the bars
text(gender_dist,gender/2,labels=gender,pos=3)

# Identify near-zero variance predictors for categorical variables
categorical<-sapply(pre_data,is.factor)
categorical
nzv_result <- nearZeroVar(pre_data[,categorical], saveMetrics = TRUE)
print(nzv_result)

#Class distribution
pre_data

## Missing values
sapply(pre_data, function(x) sum(is.na(x)))

## Frequency distribution - Skewness
par(mfrow= c(3,4))
for(i in 1:11){
  hist(pre_data[,i], main = names(pre_data[i]), xlab=names(pre_data[i]))
  
}

#Skewness
skewValues <- apply(pre_data[1:11], 2, skewness)
skewValues

## Outliers
par(mfrow= c(3,4))
for(i in 1:11){
  boxplot(pre_data[,i], main = names(pre_data[i]))
}

cor_data <- cor(pre_data[c(1,2,3,4,5,6,7,8,9,10,11)])
cor_data
par(mfrow= c(1,1))
corrplot(cor_data)

## Box-Cox transformation
xx1 <- preProcess(pre_data[1:11], method = c("BoxCox", "spatialSign","center","scale"))
xx1

# Apply the transformations:
transformed <- predict(xx1, pre_data[1:11])
apply(pre_data[1:11],2,skewness)
apply(transformed,2,skewness)
colnames(transformed)

#hist plots after transformation
par(mfrow= c(3,4))
for(i in 1:11){
  hist(transformed[,i], main = names(transformed[i]), xlab= names(transformed[i]),col =
         "grey")
}

#Barplot after transformation
par(mfrow= c(3,4))
for(i in 1:11){
  boxplot(transformed[,i], main = names(transformed[i]))
}

#Removing highly correlated data
highCorr <- findCorrelation(cor_data, cutoff = .75)
length(highCorr)
highCorr
filteredSegData <- transformed[, -highCorr]
filteredSegData

#Correlation plot after removing highly correlated variables
par(mfrow=c(1,1))
corrplot(cor(filteredSegData))
filteredSegData <- cbind(filteredSegData ,pre_data$gender_F)
colnames(filteredSegData)[11] <- "gender"

#Split data into train and test sets
trainingRows<-createDataPartition(pre_data[,12],p=0.8,list = FALSE)
trainX<-filteredSegData[trainingRows,]
trainY<-as.factor(pre_data[trainingRows,12])
testX<-filteredSegData[-trainingRows,]
testY<-as.factor(pre_data[-trainingRows,12])


###MODELS BUILDING###
##Linear Classification Models##
#Logistic Regression#
set.seed(100)
ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = defaultSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
lrFull <- train(trainX,
                y = trainY,
                method = "multinom",
                metric = "Kappa",
                trControl = ctrl)
lrFull
plot(lrFull)
summary(lrFull)
lrPred <- predict(lrFull,newdata = testX)
confusionMatrix(lrPred,testY)

#Linear Discriminant Analysis#
LDAFull <- train(trainX,
                 y = trainY,
                 method = "lda",
                 metric = "Kappa",
                 
                 trControl = ctrl)
LDAFull
plot(LDAFull)
summary(LDAFull)
LDAPred <- predict(LDAFull,newdata = testX)
confusionMatrix(LDAPred,testY)

#Partial Least Squares Discriminant Analysis#
set.seed(476)
ctrl <- trainControl(summaryFunction = defaultSummary,
                     classProbs = TRUE)
plsFit <- train(x = trainX,
                y = trainY,
                method = "pls",
                tuneGrid = expand.grid(.ncomp = 1:10),
                preProc = c("center","scale"),
                metric = "Kappa",
                trControl = ctrl)
plsFit
plot(plsFit)
#summary(plsFit)
plsPred <- predict(plsFit,newdata = testX)
confusionMatrix(plsPred,testY)

#glmnet#
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
set.seed(476)
glmnTuned <- train(x=trainX,
                   y = trainY,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "Kappa",
                   trControl = ctrl)
glmnTuned
plot(glmnTuned)
summary(glmnTuned)
glmnPred <- predict(glmnTuned,newdata = testX)
confusionMatrix(glmnPred,testY)

##varImp(plsFit)
##plot(varImp(plsFit),top=5)

# Nearest Shrunken Centroids model
library(pamr)
nsc_Grid <- data.frame(.threshold=seq(0,4, by=0.1))
set.seed(951)
nsc_model <- train(x=trainX,y=trainY,
                   method = "pam",
                   tuneGrid = nsc_Grid,
                   metric = "Kappa",
                   trControl = ctrl)
nsc_model
pred_nsc_model <- predict(nsc_model, testX)
pred_nsc_model
plot(nsc_model)
summary(nsc_model)
confusionMatrix(data = pred_nsc_model,reference = testY)


##NON-LINEAR CLASSIFICATION MODELS##
#Mixture Discriminantion Analysis#
set.seed(100)
ctrl <- trainControl(summaryFunction = defaultSummary,
                     classProbs = TRUE)
mdaFit <- train(x = trainX,
                y = trainY,
                method = "mda",
                metric = "Kappa",
                tuneGrid = expand.grid(.subclasses = 1:10),
                trControl = ctrl)
mdaFit
plot(mdaFit)
summary(mdaFit)
mdaPred <- predict(mdaFit,newdata = testX)
confusionMatrix(mdaPred,testY)
mdaFit <- train(x = trainX,
                y = trainY,
                method = "mda",
                metric = "Kappa",
                tuneGrid = expand.grid(.subclasses = 1:4),
                trControl = ctrl)
mdaFit
plot(mdaFit)
summary(mdaFit)
mdaPred <- predict(mdaFit,newdata = testX)
confusionMatrix(mdaPred,testY)

# Regularized Discriminant Analysis (RDA)
rdaGrid <- expand.grid(.gamma= 1:10, .lambda = c(0, .1, 1, 2))
rdaFit <- train(x = trainX,
                y = trainY,
                method = "rda",
                metric = "Kappa",
                tuneGrid = rdaGrid,
                trControl = ctrl)
rdaFit
plot(rdaFit)
summary(rdaFit)
pred_rda_model <- predict(rdaFit,newdata = testX)
confusionMatrix(pred_rda_model,testY)

# Quadratic Discriminant Analysis (QDA)
set.seed(709)
ctrl<- trainControl(method = "cv", number = 5, returnResamp = "all",
                    classProbs = TRUE,
                    summaryFunction = defaultSummary)
tuneGrid <- expand.grid(parameter = seq(0, 1, by = 0.1))
qdaFit <- train(x = trainX,
                y = trainY,
                method = "qda",
                metric = "Kappa",
                tuneGrid = data.frame(),
                preProcess = c("center","scale"),
                trControl = ctrl)
qdaFit
plot(qdaFit)
summary(qdaFit)
qdaPred <- predict(qdaFit,newdata = testX)
confusionMatrix(qdaPred,testY)

#Neural Networks#
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (11 + 1) + (maxSize+1)*2)
ctrl <- trainControl(summaryFunction = defaultSummary,
                     classProbs = TRUE)
set.seed(456)
nnetFit <- train(x = trainX,
                 y = trainY,
                 method = "nnet",
                 metric = "Kappa",
                 preProc = c("center", "scale", "spatialSign"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit
plot(nnetFit)
summary(nnetFit)
nnetPred <- predict(nnetFit,newdata = testX)
confusionMatrix(nnetPred,testY)

#Flexible Discriminant Analysis#
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
fdaTuned <- train(x = trainX,
                  y = trainY,
                  method = "fda",
                  metric = "Kappa",
                  # Explicitly declare the candidate models to test
                  tuneGrid = marsGrid,
                  trControl = trainControl(method = "cv"))
fdaTuned
plot(fdaTuned)
summary(fdaTuned)
fdaPred <- predict(fdaTuned,newdata = testX)
confusionMatrix(fdaPred,testY)

#Support Verctor Machine#
ctrl <- trainControl(summaryFunction = defaultSummary,
                     classProbs = TRUE)
sigmaRangeReduced <- sigest(as.matrix(trainX))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 10)))
set.seed(476)
svmRModel <- train(x = trainX,
                   y = trainY,
                   method = "svmRadial",
                   metric = "Kappa",
                   preProc = c("center", "scale"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = ctrl)
svmRModel
plot(svmRModel)
ggplot(svmRModel)+coord_trans(x='log2')
summary(svmRModel)
svmPred <- predict(svmRModel,newdata = testX)
confusionMatrix(svmPred,testY)

#K-Nearest Neighbours#
ctrl <- trainControl(summaryFunction = defaultSummary,
                     classProbs = TRUE)
set.seed(476)
knnFit <- train(x = trainingData,
                y = trainResponse,
                method = "knn",
                metric = "Kappa",
                preProc = c("center", "scale"),
                ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is
                the best
                tuneGrid = data.frame(.k = 1:50),
                trControl = ctrl)
knnFit
plot(knnFit)
summary(knnFit)
knnPred <- predict(knnFit,newdata = testingData)
confusionMatrix(knnPred,testResponse)

#Bayesian#
set.seed(476)
nbFit <- train( x = trainingData,
                y = trainResponse,
                method = "nb",
                metric = "Kappa",
                ## preProc = c("center", "scale"),
                ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is
                the best
                tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
                trControl = ctrl)
nbFit
plot(nbFit)
summary(nbFit)
nbPred <- predict(nbFit,newdata = testingData)
confusionMatrix(nbPred,testResponse)

#Important variables for best model
# Extract variable importance scores
top_5 <- varImp(nnetFit)$importance

# Subset the matrix to include only rows where "Overall" > 0
top_5_subset <- top_5[top_5[, "Overall"] > 0, ]

# Sort the data frame in descending order based on the "Overall" column
top_5_subset <- top_5_subset[order(-top_5_subset[, "Overall"]), ]

# Plot the top 5 important variables
if (nrow(top_5_subset) > 0) {
  barplot(top_5_subset[, "Overall"], main = "Top Important Variables", col = "grey", las =
            2)
} else {
  cat("No variables with positive importance scores.\n")
}
top5 <- varImp(svmRModel)
Top5
plot(top5)