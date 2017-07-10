### House Prices: Advanced Regression Techniques
# Set workdir
setwd("/Users/sophieyang/Documents/Kaggle/HousePrices")

# Set seed
set.seed(1)

# Load Libraries
library(randomForest)
library(caret)
library(neuralnet)
library(MASS)
library(Metrics)
library(ModelMetrics)
library(e1071)
library(xgboost)

# Get files
train <- read.csv("trainClean.csv")
test <- read.csv("testClean.csv")
test <- test[names(test) != "is_train"]
dataFinal <- read.csv("traintestClean.csv")
dataNoId <- dataFinal[names(dataFinal) != "Id"]

# Set as appropriate (factor, num, etc)
train$SalePrice <- as.numeric(train$SalePrice)
train$BsmtFinSF1 <- as.numeric(train$BsmtFinSF1)
train$BsmtFinSF2 <- as.numeric(train$BsmtFinSF2)
train$BsmtUnfSF <- as.numeric(train$BsmtUnfSF)
train$TotalBsmtSF <- as.numeric(train$TotalBsmtSF)
train$LowQualFinSF <- as.numeric(train$LowQualFinSF)
train$WoodDeckSF <- as.numeric(train$WoodDeckSF)
train$OpenPorchSF <- as.numeric(train$OpenPorchSF)
train$GarageCars <- as.numeric(train$GarageCars)
train$GarageArea <- as.numeric(train$GarageArea)
train$BsmtFullBath <- as.numeric(train$BsmtFullBath)
train$BsmtHalfBath <- as.numeric(train$BsmtHalfBath)
train$MasVnrArea <- as.numeric(train$MasVnrArea)
train$GarageYrBlt <- as.numeric(train$GarageYrBlt)
train$LotFrontage <- as.numeric(train$LotFrontage)
train$MSSubClass <- as.factor(train$MSSubClass)
train$LotArea <- as.numeric(train$LotArea)
train$OverallQual <- as.factor(train$OverallQual)
train$OverallCond <- as.factor(train$OverallCond)
train$YearBuilt <- as.numeric(train$YearBuilt)
train$YearRemodAdd <- as.numeric(train$YearRemodAdd)
train$X1stFlrSF <- as.numeric(train$X1stFlrSF)
train$X2ndFlrSF <- as.numeric(train$X2ndFlrSF)
train$GrLivArea <- as.numeric(train$GrLivArea)
train$FullBath <- as.numeric(train$FullBath)
train$HalfBath <- as.numeric(train$HalfBath)
train$BedroomAbvGr <- as.numeric(train$BedroomAbvGr)
train$KitchenAbvGr <- as.numeric(train$KitchenAbvGr)
train$TotRmsAbvGrd <- as.numeric(train$TotRmsAbvGrd)
train$Fireplaces <- as.numeric(train$Fireplaces)
train$EnclosedPorch <- as.numeric(train$EnclosedPorch)
train$X3SsnPorch <- as.numeric(train$X3SsnPorch)
train$ScreenPorch <- as.numeric(train$ScreenPorch)
train$PoolArea <- as.numeric(train$PoolArea)
train$MiscVal <- as.numeric(train$MiscVal)
train$MoSold <- as.factor(train$MoSold)
train$YrSold <- as.numeric(train$YrSold)

test$BsmtFinSF1 <- as.numeric(test$BsmtFinSF1)
test$BsmtFinSF2 <- as.numeric(test$BsmtFinSF2)
test$BsmtUnfSF <- as.numeric(test$BsmtUnfSF)
test$TotalBsmtSF <- as.numeric(test$TotalBsmtSF)
test$LowQualFinSF <- as.numeric(test$LowQualFinSF)
test$WoodDeckSF <- as.numeric(test$WoodDeckSF)
test$OpenPorchSF <- as.numeric(test$OpenPorchSF)
test$GarageCars <- as.numeric(test$GarageCars)
test$GarageArea <- as.numeric(test$GarageArea)
test$BsmtFullBath <- as.numeric(test$BsmtFullBath)
test$BsmtHalfBath <- as.numeric(test$BsmtHalfBath)
test$MasVnrArea <- as.numeric(test$MasVnrArea)
test$GarageYrBlt <- as.numeric(test$GarageYrBlt)
test$LotFrontage <- as.numeric(test$LotFrontage)
test$MSSubClass <- as.factor(test$MSSubClass)
test$LotArea <- as.numeric(test$LotArea)
test$OverallQual <- as.factor(test$OverallQual)
test$OverallCond <- as.factor(test$OverallCond)
test$YearBuilt <- as.numeric(test$YearBuilt)
test$YearRemodAdd <- as.numeric(test$YearRemodAdd)
test$X1stFlrSF <- as.numeric(test$X1stFlrSF)
test$X2ndFlrSF <- as.numeric(test$X2ndFlrSF)
test$GrLivArea <- as.numeric(test$GrLivArea)
test$FullBath <- as.numeric(test$FullBath)
test$HalfBath <- as.numeric(test$HalfBath)
test$BedroomAbvGr <- as.numeric(test$BedroomAbvGr)
test$KitchenAbvGr <- as.numeric(test$KitchenAbvGr)
test$TotRmsAbvGrd <- as.numeric(test$TotRmsAbvGrd)
test$Fireplaces <- as.numeric(test$Fireplaces)
test$EnclosedPorch <- as.numeric(test$EnclosedPorch)
test$X3SsnPorch <- as.numeric(test$X3SsnPorch)
test$ScreenPorch <- as.numeric(test$ScreenPorch)
test$PoolArea <- as.numeric(test$PoolArea)
test$MiscVal <- as.numeric(test$MiscVal)
test$MoSold <- as.factor(test$MoSold)
test$YrSold <- as.numeric(test$YrSold)

##################################################################################
#### 2. Remove outliers
##################################################################################

### Perform exploratory analysis with training data
trainExp <- train
# Remove training points with LotArea > 56600 (60000)
trainExp <- droplevels(subset(trainExp, trainExp$LotArea <= 60000))
# Remove a few outliers with yearsbuilt 
trainExp <- droplevels(subset(trainExp, (trainExp$YearBuilt > 1900) | (trainExp$SalePrice < 200000)))
# NEW PREDICTOR: Remodeled -> whether property has been remodeled
remodeled <- trainExp$YearRemodAdd != trainExp$YearBuilt
trainExp <- data.frame(trainExp, remodeled)
# Remove heating = OthW, Floor from training set
trainExp <- droplevels(subset(trainExp, (trainExp$Heating != "OthW") & (trainExp$Heating != "Floor")))
# NEW PREDICTOR: Bangalow <- X2ndFlrSF == 0
Bangalow <- trainExp$X2ndFlrSF == 0
trainExp <- data.frame(trainExp, Bangalow)
# plot -> plot(y=trainExp$SalePrice, x=trainExp$Bangalow)
# Remove outliers for GrLivArea
# plot -> plot(y=trainExp$SalePrice, x=trainExp$GrLivArea)
trainExp <- droplevels(subset(trainExp, (trainExp$GrLivArea < 3300) | (trainExp$SalePrice > 210000)))
# Remove outlier for FullBath
trainExp <- droplevels(subset(trainExp, (trainExp$FullBath > 0.5) | (trainExp$SalePrice < 300000)))
# NEW PREDICTOR: Wooddeck <- WoodDeckSF > 0
Wooddeck <- trainExp$WoodDeckSF == 0
trainExp <- data.frame(trainExp, Wooddeck)
# plot -> plot(y=trainExp$SalePrice, x=trainExp$Wooddeck)
# NEW PREDICTOR: Pool <- PoolArea > 0
Pool <- trainExp$PoolArea > 0
trainExp <- data.frame(trainExp, Pool)
# Remove outlets in Yr Sold
trainExp <- droplevels(subset(trainExp, (trainExp$YrSold != 2006) | (trainExp$SalePrice < 500000)))
trainExp <- droplevels(subset(trainExp, (trainExp$YrSold != 2007) | (trainExp$SalePrice < 500000)))
trainExp <- droplevels(subset(trainExp, (trainExp$YrSold != 2009) | (trainExp$SalePrice < 450000)))
trainExp <- droplevels(subset(trainExp, (trainExp$YrSold != 2010) | (trainExp$SalePrice < 500000)))
# NEW PREDICTOR: Bsmt <- TotalBsmtSF > 0
Bsmt <- trainExp$TotalBsmtSF > 0
trainExp <- data.frame(trainExp, Bsmt)
# Remove 3 outliers in TotalBSmtSF
trainExp <- droplevels(subset(trainExp, trainExp$TotalBsmtSF < 3000)) # drops 2
trainExp <- droplevels(subset(trainExp, (trainExp$TotalBsmtSF < 2000) | (trainExp$SalePrice > 150000))) # drops 1
# Remove 2 outliers in GarageCars
trainExp <- droplevels(subset(trainExp, (trainExp$GarageCars > 1.5) | (trainExp$SalePrice < 280000))) # drops 2
# Remove 4 outliers in GarageArea
trainExp <- droplevels(subset(trainExp, (trainExp$GarageArea < 1200) | (trainExp$SalePrice > 300000))) # drops 3
trainExp <- droplevels(subset(trainExp, (trainExp$GarageArea < 1000) | (trainExp$SalePrice > 150000))) # drops 1
# Remove 2 outliers in GrLivArea (already removed)
# Remove points where Utilities is "NoSeWa"
trainExp <- droplevels(subset(trainExp, trainExp$Utilities == "AllPub"))
# Remove Utilities
trainExp <- trainExp[names(trainExp) != "Utilities"]

trainExp$remodeled <- as.factor(trainExp$remodeled)
trainExp$Bangalow <- as.factor(trainExp$Bangalow)
trainExp$Wooddeck <- as.factor(trainExp$Wooddeck)
trainExp$Pool <- as.factor(trainExp$Pool)
trainExp$Bsmt <- as.factor(trainExp$Bsmt)

### Perform exploratory analysis with testing data
testExp <- test
# NEW PREDICTOR: Remodeled -> whether property has been remodeled
remodeled <- testExp$YearRemodAdd != testExp$YearBuilt
testExp <- data.frame(testExp, remodeled)
# NEW PREDICTOR: Bangalow <- X2ndFlrSF == 0
Bangalow <- testExp$X2ndFlrSF == 0
testExp <- data.frame(testExp, Bangalow)
# NEW PREDICTOR: Wooddeck <- WoodDeckSF == 0
Wooddeck <- testExp$WoodDeckSF > 0
testExp <- data.frame(testExp, Wooddeck)
# NEW PREDICTOR: Pool <- PoolArea > 0
Pool <- testExp$PoolArea > 0
testExp <- data.frame(testExp, Pool)
# NEW PREDICTOR: Bsmt <- TotalBsmtSF > 0
Bsmt <- testExp$TotalBsmtSF > 0
testExp <- data.frame(testExp, Bsmt)
# Remove Utilities
testExp <- testExp[names(testExp) != "Utilities"]

testExp$remodeled <- as.factor(testExp$remodeled)
testExp$Bangalow <- as.factor(testExp$Bangalow)
testExp$Wooddeck <- as.factor(testExp$Wooddeck)
testExp$Pool <- as.factor(testExp$Pool)
testExp$Bsmt <- as.factor(testExp$Bsmt)

##################################################################################
#### 3a. Find high skew continuous data, and transform via log
##################################################################################

# Check the skewness of the continuous data:
skew <- as.numeric()
name <- as.character()

for (i in 1:ncol(train)) {
  if(is.numeric(train[,i])) {
    skew <- c(skew, skewness(train[,i]))
    name <- c(name, names(train)[i])
  }
}

skew.df <- data.frame(name = name, skew = skew)
skew.df <- skew.df[order(skew.df$skew, decreasing = TRUE),]
skewLarge.df <- subset(skew.df, (abs(skew.df$skew) > 3) ) # & (skew.df$name != "SalePrice"))

# Transform columns with high skew via logarithm or scaling
# It seems like log(x+1) has a better effect on the 

trainExp[which(names(trainExp) %in% skewLarge.df$name)] <- log(1+trainExp[which(names(trainExp) %in% skewLarge.df$name)])
testExp[which(names(testExp) %in% skewLarge.df$name)] <- log(1+testExp[which(names(testExp) %in% skewLarge.df$name)])

# plot(y=trainExp$SalePrice, x=trainExp$YrSold)
# histogram(dataFinal$Id)

##################################################################################
#### 3b. Encode categorical variables to numerical or bins or use dummy variables
##################################################################################

# Attach train and test together to extract dummy vars
traintestExp <- rbind(trainExp, testExp)

#trainDummy <- dummyVars(~ ., data = trainExp, fullRank = TRUE)
#trainBin <- data.frame(predict(trainDummy, newdata = trainExp))
#testDummy <- dummyVars(~ ., data = testExp, fullRank = TRUE)
#testBin <- data.frame(predict(testDummy, newdata = testExp))

traintestDummy <- dummyVars(~ ., data = traintestExp, fullRank = TRUE)
traintestBin <- data.frame(predict(traintestDummy, newdata = traintestExp))

trainBin <- traintestBin[1:nrow(trainExp),]
testBin <- traintestBin[(nrow(trainExp)+1):nrow(traintestBin),]

#names(testBin)[which(!(names(testBin) %in% names(trainBin)))]
#names(trainBin)[which(!(names(trainBin) %in% names(testBin)))]

##################################################################################
#### 3c. Find Correlation and remove
##################################################################################
# Print out training values with only 1 type and remove from training and test data
predBinUnique <- as.character()

for (j in 1:ncol(trainBin)) {
  len <- length(unique(trainBin[,j]))
  if(len < 2) {
    predBinUnique <- c(predBinUnique, names(trainBin)[j])
  }
}

trainBin <- trainBin[!(names(trainBin) %in% predBinUnique)]
testBin <- testBin[!(names(testBin) %in% predBinUnique)]

corrNum <- cor(trainBin, trainBin)

corrThresh <- 0.2

pred1 <- as.character()
pred2 <- as.character()
corrCoef <- as.numeric()

for (i in 1:(nrow(corrNum)-1)) {
  len <- length(corrNum[i:nrow(corrNum),i][(abs(corrNum[i:nrow(corrNum),i])>corrThresh)])
  if(len > 0) {
    for(k in 1:len) {
      pred1 <- c(pred1, rownames(corrNum)[i])
      pred2 <- c(pred2, names(corrNum[i:nrow(corrNum),i][is.na(corrNum[i:nrow(corrNum),i]) | (abs(corrNum[i:nrow(corrNum),i])>corrThresh)])[k])
      corrCoef <- c(corrCoef, corrNum[i:nrow(corrNum),i][is.na(corrNum[i:nrow(corrNum),i]) | (abs(corrNum[i:nrow(corrNum),i])>corrThresh)][k])
    }
  }
}
length(pred1)
length(pred2)
length(corrCoef)

corrNum.df <- data.frame(pred1 = pred1, pred2 = pred2, corrCoef = corrCoef)
corrNum.df <- corrNum.df[order(abs(corrNum.df$corrCoef),decreasing = TRUE),]
corrNum.df <- subset(corrNum.df, as.character(corrNum.df$pred1) != as.character(corrNum.df$pred2))

sum(is.na(corrNum.df$corrCoef))
charPred <- as.character(corrNum.df[which(corrNum.df$corrCoef == 1),]$pred2)

# Remove "BldgType.Duplex", "Exterior2nd.CBlock"
trainBin <- trainBin[!(names(trainBin) %in% charPred)]
testBin <- testBin[!(names(testBin) %in% charPred)]

##################################################################################
#### 4. Separate into 4 folds
##################################################################################
set.seed(56)
# Randomize the training selection first
trainRandom <- trainBin[sample(nrow(trainBin)),]

# log trainsform SalePrice
#trainRandomOrig <- trainRandom
#trainRandom$SalePrice <- log(trainRandom$SalePrice)

# Use 10% for training testing data, and the rest for cross validating the model
inTrain <- createDataPartition(y = trainRandom$SalePrice, p = 0.9, list = FALSE)
trainTrain <- trainRandom[inTrain,]
trainTest <- trainRandom[-inTrain,]

all_levels <- TRUE
k <- 0
for (i in 1:length(trainTrain)) {
    if (length(unique(trainTrain[,i])) < 2) {
      all_levels <- FALSE
      k <- i
      break
    }
}
all_levels

# 1423 samples divide into 5 different sampling of 284, 284, 285, 285, 285 for CV 
# foldE <- 284
# foldO <- 285
# trainTestFold1 <- trainRandom[1:foldE,]
# trainTrainFold1 <- trainRandom[(foldE+1):nrow(trainRandom),]
# trainTestFold2 <- trainRandom[(foldE+1):(2*foldE),]
# trainTrainFold2 <- rbind(trainRandom[1:foldE,], trainRandom[(2*foldE+1):nrow(trainRandom),])
# trainTestFold3 <- trainRandom[(2*foldE+1):(2*foldE+foldO),]
# trainTrainFold3 <- rbind(trainRandom[1:(2*foldE),], trainRandom[(2*foldE+foldO+1):nrow(trainRandom),])
# trainTestFold4 <- trainRandom[(2*foldE+foldO+1):(2*foldE+2*foldO),]
# trainTrainFold4 <- rbind(trainRandom[1:(2*foldE+foldO),], trainRandom[(2*foldE+2*foldO+1):nrow(trainRandom),])
# trainTestFold5 <- trainRandom[(2*foldE+2*foldO+1):nrow(trainRandom),]
# trainTrainFold5 <- trainRandom[1:(2*foldE+2*foldO),]

# 1423 samples divide into 5 different sampling of 355, 356, 356, 356

#foldE <- 355
#foldO <- 356
#trainTestFold1 <- trainRandom[1:foldE,]
#trainTrainFold1 <- trainRandom[(foldE+1):nrow(trainRandom),]
#trainTestFold2 <- trainRandom[(foldE+1):(foldE+foldO),]
#trainTrainFold2 <- rbind(trainRandom[1:foldE,], trainRandom[(foldE+foldO+1):nrow(trainRandom),])
#trainTestFold3 <- trainRandom[(foldE+foldO+1):(foldE+2*foldO),]
#trainTrainFold3 <- rbind(trainRandom[1:(foldE+foldO),], trainRandom[(foldE+2*foldO+1):nrow(trainRandom),])
#trainTestFold4 <- trainRandom[(foldE+2*foldO+1):nrow(trainRandom),]
#trainTrainFold4 <- trainRandom[1:(foldE+2*foldO),]

# 1423 samples divide into 10 samples 142*9+145
foldE <- 142
trainTestFold1 <- trainRandom[1:foldE,]
trainTrainFold1 <- trainRandom[(foldE+1):nrow(trainRandom),]
trainTestFold2 <- trainRandom[(foldE+1):(2*foldE),]
trainTrainFold2 <- rbind(trainRandom[1:foldE,], trainRandom[(2*foldE+1):nrow(trainRandom),])
trainTestFold3 <- trainRandom[(2*foldE+1):(3*foldE),]
trainTrainFold3 <- rbind(trainRandom[1:(2*foldE),], trainRandom[(3*foldE+1):nrow(trainRandom),])
trainTestFold4 <- trainRandom[(3*foldE+1):(4*foldE),]
trainTrainFold4 <- rbind(trainRandom[1:(3*foldE),], trainRandom[(4*foldE+1):nrow(trainRandom),])
trainTestFold5 <- trainRandom[(4*foldE+1):(5*foldE),]
trainTrainFold5 <- rbind(trainRandom[1:(4*foldE),], trainRandom[(5*foldE+1):nrow(trainRandom),])
trainTestFold6 <- trainRandom[(5*foldE+1):(6*foldE),]
trainTrainFold6 <- rbind(trainRandom[1:(5*foldE),], trainRandom[(6*foldE+1):nrow(trainRandom),])
trainTestFold7 <- trainRandom[(6*foldE+1):(7*foldE),]
trainTrainFold7 <- rbind(trainRandom[1:(6*foldE),], trainRandom[(7*foldE+1):nrow(trainRandom),])
trainTestFold8 <- trainRandom[(7*foldE+1):(8*foldE),]
trainTrainFold8 <- rbind(trainRandom[1:(7*foldE),], trainRandom[(8*foldE+1):nrow(trainRandom),])
trainTestFold9 <- trainRandom[(8*foldE+1):(9*foldE),]
trainTrainFold9 <- rbind(trainRandom[1:(8*foldE),], trainRandom[(9*foldE+1):nrow(trainRandom),])
trainTestFold10 <- trainRandom[(9*foldE+1):nrow(trainRandom),]
trainTrainFold10 <- trainRandom[1:(9*foldE),]


##################################################################################
#### Try various models:
### Random Forest, Gradient Boosting, Extreme gradient boosting, SVM regressor
### Ridge, Lasso
##################################################################################

# To test the error rate, the Leaderboard uses log RMSLE, which uses the following equation:
myRMSLE <- function(actual, predicted) {
  sqrt(1/(length(actual)) * sum((log(actual)-log(predicted))^2))
}

myRMSE <- function(actual, predicted) {
  sqrt(1/(length(actual)) * sum(((actual)-(predicted))^2))
}

##################################################################################
#### 5. Random Forest:
##################################################################################

set.seed(56)

# Set up trainControl for CV
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)

# Run RF model
RFmodel <- train(SalePrice ~ .,
                 method = "rf",
                 trControl = ctrl,
                 preProc = c("center", "scale"),
                 data = trainTrain[,-1]
)

RFpred.trainTrain <- predict(RFmodel, newdata = data.frame(trainTrain))
RFpred.trainTest <- predict(RFmodel, newdata = data.frame(trainTest))

myRMSLE(actual = trainTrain$SalePrice, predicted = RFpred.trainTrain)
# 0.06303653
myRMSLE(actual = trainTest$SalePrice, predicted = RFpred.trainTest)
# 0.1323178

##########
# Fold 10
##########
set.seed(56)
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
feat_remove <- c("Id")
RFmodel <- train(SalePrice ~ .,
                 method = "rf",
                 trControl = ctrl,
                 preProc = c("center", "scale"),
                 data = trainTrainFold1[!(names(trainTrainFold1) %in% feat_remove)]
)
RFpred.trainTrainFold1 <- predict(RFmodel, newdata = data.frame(trainTrainFold1))
RFpred.trainTestFold1 <- predict(RFmodel, newdata = data.frame(trainTestFold1))

myRMSLE(actual = trainTrainFold1$SalePrice, predicted = RFpred.trainTrainFold1)
# 0.06363821
myRMSLE(actual = trainTestFold1$SalePrice, predicted = RFpred.trainTestFold1)
# 0.1396616

#############################
# Predict actual test values
#############################
RFpred.test <- predict(RFmodel, newdata = data.frame(testBin))

##################################################################################
#### 6. Gradient Boosting:
##################################################################################

set.seed(56)

# Set up trainControl for CV
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 4, verboseIter = TRUE)

# Run model
GBMmodel <- train(SalePrice ~ .,
                  method = "gbm",
                  trControl = ctrl,
                  preProc = c("center", "scale"),
                  data = trainTrain[,-1]
)

GBMpred.trainTrain <- predict(GBMmodel, newdata = data.frame(trainTrain))
GBMpred.trainTest <- predict(GBMmodel, newdata = data.frame(trainTest))

myRMSLE(actual = trainTrain$SalePrice, predicted = GBMpred.trainTrain)
# 0.1072543
myRMSLE(actual = trainTest$SalePrice, predicted = GBMpred.trainTest)
# 0.1102761

#########################
# Folds 10
#########################
set.seed(56)
ctrl <- trainControl(method = "repeatedcv", repeats = 2, number = 4, verboseIter = TRUE)
feat_remove <- c("Id")
GBMmodel <- train(SalePrice ~ .,
                  method = "gbm",
                  trControl = ctrl,
                  preProc = c("center", "scale"),
                  data = trainTrainFold2[!(names(trainTrainFold2) %in% feat_remove)]
)

GBMpred.trainTrainFold2 <- predict(GBMmodel, newdata = data.frame(trainTrainFold2))
GBMpred.trainTestFold2 <- predict(GBMmodel, newdata = data.frame(trainTestFold2))

myRMSLE(actual = trainTrainFold2$SalePrice, predicted = GBMpred.trainTrainFold2)
# 0.106961
myRMSLE(actual = trainTestFold2$SalePrice, predicted = GBMpred.trainTestFold2)
# 0.110437


# Predict actual test values
GBMpred.test <- predict(GBMmodel, newdata = data.frame(testBin))

##################################################################################
### 7. Linear Regression:
##################################################################################
# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 8, number = 5, verboseIter = TRUE)
# 
# # Run model
# feat_remove <- c("Id")
# LMmodel <- train(SalePrice ~ .,
#                   method = "lm",
#                   trControl = ctrl,
#                   preProc = c("center", "scale"),
#                   data = trainTrain[!(names(trainTrain) %in% feat_remove)]
# )
# 
# LMpred.trainTrain <- predict(LMmodel, newdata = data.frame(trainTrain))
# LMpred.trainTest <- predict(LMmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = LMpred.trainTrain)
# # 0.09924477
# myRMSLE(actual = trainTest$SalePrice, predicted = LMpred.trainTest)
# # 0.1256265

################
# Folds 10
################
set.seed(56)
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 4, verboseIter = TRUE)
feat_remove <- c("Id","Neighborhood.Blueste", "Condition2.RRAn")
LMmodel <- train(SalePrice ~ .,
                 method = "lm",
                 trControl = ctrl,
                 preProc = c("center", "scale"),
                 data = trainTrainFold3[!(names(trainTrainFold3) %in% feat_remove)]
)

LMpred.trainTrainFold3 <- predict(LMmodel, newdata = data.frame(trainTrainFold3))
LMpred.trainTestFold3 <- predict(LMmodel, newdata = data.frame(trainTestFold3))

myRMSLE(actual = trainTrainFold3$SalePrice, predicted = LMpred.trainTrainFold3)
# 0.09804128
myRMSLE(actual = trainTestFold3$SalePrice, predicted = LMpred.trainTestFold3)
# 0.1441017



# # Locally stack LM and GBM
# trainTrainLM_X_GBM <- data.frame(trainTrain, LMpred = LMpred.trainTrain, GBMpred = GBMpred.trainTrain)
# trainTestLM_X_GBM <- data.frame(trainTest, LMpred = LMpred.trainTest, GBMpred = GBMpred.trainTest)
# 
# LM_X_GBMmodel <- train(SalePrice ~ .,
#                        method = "lm",
#                        trControl = ctrl,
#                        preProc = c("center", "scale"),
#                        data = trainTrainLM_X_GBM[,-1]
# )
# 
# LM_X_GBMpred.trainTrain <- predict(LM_X_GBMmodel, newdata = data.frame(trainTrainLM_X_GBM))
# LM_X_GBMpred.trainTest <- predict(LM_X_GBMmodel, newdata = data.frame(trainTestLM_X_GBM))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = LM_X_GBMpred.trainTrain)
# # 0.08471641
# myRMSLE(actual = trainTest$SalePrice, predicted = LM_X_GBMpred.trainTest)
# # 0.1109008

############################
# Predict actual test values
############################
LMpred.test <- predict(LMmodel, newdata = data.frame(testBin))

##################################################################################
#### 8. Extreme Gradient Boosting: Using trees
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# 
# # Run model
# XGBTmodel <- train(SalePrice ~ .,
#                   method = "xgbTree",
#                   trControl = ctrl,
#                   preProc = c("center", "scale"),
#                   data = trainTrain[,-1]
# )
# 
# XGBTpred.trainTrain <- predict(XGBTmodel, newdata = data.frame(trainTrain))
# XGBTpred.trainTest <- predict(XGBTmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = XGBTpred.trainTrain)
# # 0.08510203
# myRMSLE(actual = trainTest$SalePrice, predicted = XGBTpred.trainTest)
# 0.1337427

##########
# Folds 10
##########
set.seed(56)
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
feat_remove <- c("Id")
XGBTmodel <- train(SalePrice ~ .,
                   method = "xgbTree",
                   trControl = ctrl,
                   preProc = c("center", "scale"),
                   data = trainTrainFold4[!(names(trainTrainFold4) %in% feat_remove)]
)

XGBTpred.trainTrainFold4 <- predict(XGBTmodel, newdata = data.frame(trainTrainFold4))
XGBTpred.trainTestFold4 <- predict(XGBTmodel, newdata = data.frame(trainTestFold4))

myRMSLE(actual = trainTrainFold4$SalePrice, predicted = XGBTpred.trainTrainFold4)
# 0.0889799
myRMSLE(actual = trainTestFold4$SalePrice, predicted = XGBTpred.trainTestFold4)
# 0.1075595


############################
# Predict actual test values
############################
XGBTpred.test <- predict(XGBTmodel, newdata = data.frame(testBin))

##################################################################################
#### 9. Extreme Gradient Boosting: Linear
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# 
# # Run model
# XGBLmodel <- train(SalePrice ~ .,
#                    method = "xgbLinear",
#                    trControl = ctrl,
#                    preProc = c("center", "scale"),
#                    data = trainTrain[,-1]
# )
# 
# XGBLpred.trainTrain <- predict(XGBLmodel, newdata = data.frame(trainTrain))
# XGBLpred.trainTest <- predict(XGBLmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = XGBLpred.trainTrain)
# # 0.002041709
# myRMSLE(actual = trainTest$SalePrice, predicted = XGBLpred.trainTest)
# # 0.1296344

#########
# Fold 10
#########
set.seed(56)
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
XGBLmodel <- train(SalePrice ~ .,
                   method = "xgbLinear",
                   trControl = ctrl,
                   preProc = c("center", "scale"),
                   data = trainTrainFold5[!(names(trainTrainFold5) %in% feat_remove)]
)

XGBLpred.trainTrainFold5 <- predict(XGBLmodel, newdata = data.frame(trainTrainFold5))
XGBLpred.trainTestFold5 <- predict(XGBLmodel, newdata = data.frame(trainTestFold5))

myRMSLE(actual = trainTrainFold5$SalePrice, predicted = XGBLpred.trainTrainFold5)
# 0.003001573
myRMSLE(actual = trainTestFold5$SalePrice, predicted = XGBLpred.trainTestFold5)
# 0.1571204

##############################
# Predict actual test values
##############################
XGBLpred.test <- predict(XGBLmodel, newdata = data.frame(testBin))

##################################################################################
#### 10. Support Vector Machine: Radial Basis Function
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# 
# # Run model
# SVMRmodel <- train(SalePrice ~ .,
#                    method = "svmRadial",
#                    trControl = ctrl,
#                    preProc = c("center", "scale"),
#                    data = trainTrain[,-1]
# )
# 
# SVMRpred.trainTrain <- predict(SVMRmodel, newdata = data.frame(trainTrain))
# SVMRpred.trainTest <- predict(SVMRmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = SVMRpred.trainTrain)
# # 0.3819553
# myRMSLE(actual = trainTest$SalePrice, predicted = SVMRpred.trainTest)
# # 0.3794084
# 
# # Predict actual test values
# SVMRpred.test <- predict(SVMRmodel, newdata = data.frame(testBin))

##################################################################################
#### 11. Support Vector Machine: L2 Regularized Linear Kernel
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# 
# # Run model
# SVML2model <- train(SalePrice ~ .,
#                    method = "svmLinear3",
#                    trControl = ctrl,
#                    preProc = c("center", "scale"),
#                    data = trainTrain[,-1]
# )
# 
# SVML2pred.trainTrain <- predict(SVML2model, newdata = data.frame(trainTrain))
# SVML2pred.trainTest <- predict(SVML2model, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = SVML2pred.trainTrain)
# # 0.1031187
# myRMSLE(actual = trainTest$SalePrice, predicted = SVML2pred.trainTest)
# # 0.1208109

##########
# Fold 10
##########
set.seed(56)
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
feat_remove <- c("Id")
SVML2model <- train(SalePrice ~ .,
                    method = "svmLinear3",
                    trControl = ctrl,
                    preProc = c("center", "scale"),
                    data = trainTrainFold6[!(names(trainTrainFold6) %in% feat_remove)]
)

SVML2pred.trainTrainFold6 <- predict(SVML2model, newdata = data.frame(trainTrainFold6))
SVML2pred.trainTestFold6 <- predict(SVML2model, newdata = data.frame(trainTestFold6))

myRMSLE(actual = trainTrainFold6$SalePrice, predicted = SVML2pred.trainTrainFold6)
# 0.09980899
myRMSLE(actual = trainTestFold6$SalePrice, predicted = SVML2pred.trainTestFold6)
# 0.133366


# Predict actual test values
SVML2pred.test <- predict(SVML2model, newdata = data.frame(testBin))

##################################################################################
#### 12. Ridge Regression
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
# 
# # Run model
# feat_remove <- c("Id", 
#                  "MSSubClass.150", "Condition2.PosN", "Condition2.PosA", 
#                  "RoofMatl.Membran", "RoofMatl.Metal", "ExterCond.Po", 
#                  "Exterior1st.AsphShn", "Electrical.Mix", "MiscFeature.TenC", 
#                  "PoolQC.Fa", "Condition2.RRNn", "RoofMatl.WdShngl", 
#                  "Exterior2nd.Other", "BsmtCond.Po", "Neighborhood.Blueste")
# RIDGEmodel <- train(SalePrice ~ .,
#                    method = "ridge",
#                    trControl = ctrl,
#                    preProc = c("center", "scale"),
#                    data = trainTrain[!(names(trainTrain) %in% feat_remove)]
# )
# 
# RIDGEpred.trainTrain <- predict(RIDGEmodel, newdata = data.frame(trainTrain))
# RIDGEpred.trainTest <- predict(RIDGEmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = RIDGEpred.trainTrain)
# # 0.1001854
# myRMSLE(actual = trainTest$SalePrice, predicted = RIDGEpred.trainTest)
# # 0.1217473

#############
# Fold 10
#############
set.seed(56)
ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
feat_remove <- c("Id", "RoofMatl.Metal", "Exterior1st.ImStucc", "Exterior2nd.Other",
                 "Condition2.RRAe", "RoofStyle.Shed", "Functional.Sev",
                 "Condition1.RRNe", "Exterior1st.CBlock", "Electrical.Mix")
RIDGEmodel <- train(SalePrice ~ .,
                    method = "ridge",
                    trControl = ctrl,
                    preProc = c("center", "scale"),
                    data = trainTrainFold7[!(names(trainTrainFold7) %in% feat_remove)]
)

RIDGEpred.trainTrainFold7 <- predict(RIDGEmodel, newdata = data.frame(trainTrainFold7))
RIDGEpred.trainTestFold7 <- predict(RIDGEmodel, newdata = data.frame(trainTestFold7))

myRMSLE(actual = trainTrainFold7$SalePrice, predicted = RIDGEpred.trainTrainFold7)
# 0.1197154
myRMSLE(actual = trainTestFold7$SalePrice, predicted = RIDGEpred.trainTestFold7)
# 0.146667

#############################
# Predict actual test values
#############################
RIDGEpred.test <- predict(RIDGEmodel, newdata = data.frame(testBin))

##################################################################################
#### 13. Lasso Regression:
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
# 
# # Run model
# feat_remove <- c("Id", "MSSubClass.150","Condition2.PosA", "RoofMatl.Membran", 
#                  "RoofMatl.Metal", "ExterCond.Po", "Exterior1st.AsphShn", 
#                  "MiscFeature.TenC", "PoolQC.Fa", "Condition2.RRNn", 
#                  "RoofMatl.WdShngl", "Exterior2nd.Other", "Neighborhood.Blueste",
#                  "RoofMatl.Roll")
# LASSOmodel <- train(SalePrice ~ .,
#                     method = "lasso",
#                     trControl = ctrl,
#                     preProc = c("center", "scale"),
#                     data = trainTrain[!(names(trainTrain) %in% feat_remove)]
# )
# 
# LASSOpred.trainTrain <- predict(LASSOmodel, newdata = data.frame(trainTrain))
# LASSOpred.trainTest <- predict(LASSOmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = LASSOpred.trainTrain)
# # 0.1003812
# myRMSLE(actual = trainTest$SalePrice, predicted = LASSOpred.trainTest)
# # 0.1102661

############
## Fold 10
############
set.seed(56)
ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
feat_remove <- c("Id", "RoofMatl.Membran", "RoofMatl.Metal", "RoofMatl.Roll",
                 "Electrical.Mix", "Condition1.RRNe", "Condition2.RRAe", "HeatingQC.Po")
LASSOmodel <- train(SalePrice ~ .,
                    method = "lasso",
                    trControl = ctrl,
                    preProc = c("center", "scale"),
                    data = trainTrainFold8[!(names(trainTrainFold8) %in% feat_remove)]
)

LASSOpred.trainTrainFold8 <- predict(LASSOmodel, newdata = data.frame(trainTrainFold8))
LASSOpred.trainTestFold8 <- predict(LASSOmodel, newdata = data.frame(trainTestFold8))

myRMSLE(actual = trainTrainFold8$SalePrice, predicted = LASSOpred.trainTrainFold8)
# 0.09984732
myRMSLE(actual = trainTestFold8$SalePrice, predicted = LASSOpred.trainTestFold8)
# 0.1292511

#############################
# Predict actual test values
#############################
LASSOpred.test <- predict(LASSOmodel, newdata = data.frame(testBin))

#Id <- testBin$Id
#SalePrice <- LASSOpred.test
#output <- data.frame(Id, SalePrice)
#write.csv(output, file = "housingLassoCV.csv", row.names = FALSE)

##################################################################################
#### 14. Extra Trees
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# 
# # Run model
# ERFmodel <- train(SalePrice ~ .,
#                  method = "extraTrees",
#                  trControl = ctrl,
#                  preProc = c("center", "scale"),
#                  data = trainTrain[,-1]
# )
# 
# ERFpred.trainTrain <- predict(ERFmodel, newdata = data.frame(trainTrain))
# ERFpred.trainTest <- predict(ERFmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = ERFpred.trainTrain)
# # 0.01887272
# myRMSLE(actual = trainTest$SalePrice, predicted = ERFpred.trainTest)
# # 0.1461645
# 
# # Predict actual test values
# ERFpred.test <- predict(ERFmodel, newdata = data.frame(testBin))

##################################################################################
#### 15. K-Nearest Neighbors
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# 
# # Run model
# KNNmodel <- train(SalePrice ~ .,
#                   method = "kknn",
#                   trControl = ctrl,
#                   preProc = c("center", "scale"),
#                   data = trainTrain[,-1]
# )
# 
# KNNpred.trainTrain <- predict(KNNmodel, newdata = data.frame(trainTrain))
# KNNpred.trainTest <- predict(KNNmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = KNNpred.trainTrain)
# # 0.1295923
# myRMSLE(actual = trainTest$SalePrice, predicted = KNNpred.trainTest)
# # 0.1907924
# 
# # Predict actual test values
# KNNpred.test <- predict(KNNmodel, newdata = data.frame(testBin))

##################################################################################
#### 16. Python K-Nearest Neighbors
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# 
# # Run model
# PKNNmodel <- train(SalePrice ~ .,
#                   method = "pythonKnnReg",
#                   trControl = ctrl,
#                   preProc = c("center", "scale"),
#                   data = trainTrain[,-1]
# )
# 
# PKNNpred.trainTrain <- predict(PKNNmodel, newdata = data.frame(trainTrain))
# PKNNpred.trainTest <- predict(PKNNmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = PKNNpred.trainTrain)
# #
# myRMSLE(actual = trainTest$SalePrice, predicted = PKNNpred.trainTest)
# #
# 
# # Predict actual test values
# PKNNpred.test <- predict(PKNNmodel, newdata = data.frame(testBin))

##################################################################################
#### 17. Polynomial Regression
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# 
# # Run model
# POLYmodel <- train(SalePrice ~ .,
#                    method = "gaussprPoly",
#                    trControl = ctrl,
#                    preProc = c("center", "scale"),
#                    data = trainTrain[,-1]
# )
# 
# POLYpred.trainTrain <- predict(POLYmodel, newdata = data.frame(trainTrain))
# POLYpred.trainTest <- predict(POLYmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = POLYpred.trainTrain)
# # 0.03028932
# myRMSLE(actual = trainTest$SalePrice, predicted = POLYpred.trainTest)
# # 0.1453029
# 
# # Predict actual test values
# POLYpred.test <- predict(POLYmodel, newdata = data.frame(testBin))

##################################################################################
#### 18. ElasticNet Regression
##################################################################################
# 
# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# 
# # Run model
# feat_remove <- c("Id", "MSSubClass.150","Condition2.PosA", "RoofMatl.Membran", 
#                  "RoofMatl.Metal", "ExterCond.Po", "Exterior1st.AsphShn", 
#                  "MiscFeature.TenC", "PoolQC.Fa", "Condition2.RRNn", 
#                  "RoofMatl.WdShngl", "Exterior2nd.Other", "Neighborhood.Blueste",
#                  "RoofMatl.Roll")
# ENETmodel <- train(SalePrice ~ .,
#                    method = "enet",
#                    trControl = ctrl,
#                    preProc = c("center", "scale"),
#                    data = trainTrain[!(names(trainTrain) %in% feat_remove)]
# )
# 
# ENETpred.trainTrain <- predict(ENETmodel, newdata = data.frame(trainTrain))
# ENETpred.trainTest <- predict(ENETmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = ENETpred.trainTrain)
# # 0.1275616
# myRMSLE(actual = trainTest$SalePrice, predicted = ENETpred.trainTest)
# # 0.1124081

############
## Folds 10
############
set.seed(56)
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
feat_remove <- c("Id", "Condition2.PosN", "OverallQual.2", "RoofMatl.Metal", 
                 "ExterCond.Po", "HeatingQC.Po", "Exterior1st.AsphShn", "Functional.Sev",
                 "Condition2.RRAn", "Exterior1st.CBlock")
ENETmodel <- train(SalePrice ~ .,
                   method = "enet",
                   trControl = ctrl,
                   preProc = c("center", "scale"),
                   data = trainTrainFold9[!(names(trainTrainFold9) %in% feat_remove)]
)

ENETpred.trainTrainFold9 <- predict(ENETmodel, newdata = data.frame(trainTrainFold9))
ENETpred.trainTestFold9 <- predict(ENETmodel, newdata = data.frame(trainTestFold9))

myRMSLE(actual = trainTrainFold9$SalePrice, predicted = ENETpred.trainTrainFold9)
# 0.1015108
myRMSLE(actual = trainTestFold9$SalePrice, predicted = ENETpred.trainTestFold9)
# 0.1207192

#############################
# Predict actual test values
#############################
ENETpred.test <- predict(ENETmodel, newdata = data.frame(testBin))

# Id <- testBin$Id
# SalePrice <- ENETpred.test
# output <- data.frame(Id, SalePrice)
# write.csv(output, file = "housingENET.csv", row.names = FALSE)

##################################################################################
#### 19. Stepwise Regression
##################################################################################

# set.seed(56)
# 
# # Set up trainControl for CV
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# 
# # Run model
# feat_remove <- c("Id", "MSSubClass.150","Condition2.PosA", "RoofMatl.Membran", 
#                  "RoofMatl.Metal", "ExterCond.Po", "Exterior1st.AsphShn", 
#                  "MiscFeature.TenC", "PoolQC.Fa", "Condition2.RRNn", 
#                  "RoofMatl.WdShngl", "Exterior2nd.Other", "Neighborhood.Blueste",
#                  "RoofMatl.Roll")
# STEPmodel <- train(SalePrice ~ .,
#                    method = "leapSeq",
#                    trControl = ctrl,
#                    preProc = c("center", "scale"),
#                    data = trainTrain[!(names(trainTrain) %in% feat_remove)]
# )
# 
# STEPpred.trainTrain <- predict(STEPmodel, newdata = data.frame(trainTrain))
# STEPpred.trainTest <- predict(STEPmodel, newdata = data.frame(trainTest))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = STEPpred.trainTrain)
# # 0.2686007
# myRMSLE(actual = trainTest$SalePrice, predicted = STEPpred.trainTest)
# # 0.2821904
# 
# # Predict actual test values
# STEPpred.test <- predict(STEPmodel, newdata = data.frame(testBin))

##################################################################################
#### 20. Bayesian GLM
##################################################################################

############
## Folds 10
############
set.seed(56)
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
feat_remove <- c("Id")
BAYGmodel <- train(SalePrice ~ .,
                   method = "bayesglm",
                   trControl = ctrl,
                   preProc = c("center", "scale"),
                   data = trainTrainFold10[!(names(trainTrainFold10) %in% feat_remove)]
)

BAYGpred.trainTrainFold10 <- predict(BAYGmodel, newdata = data.frame(trainTrainFold10))
BAYGpred.trainTestFold10 <- predict(BAYGmodel, newdata = data.frame(trainTestFold10))

myRMSLE(actual = trainTrainFold10$SalePrice, predicted = BAYGpred.trainTrainFold10)
# 0.09908493
myRMSLE(actual = trainTestFold10$SalePrice, predicted = BAYGpred.trainTestFold10)
# 0.1304513

#############################
# Predict actual test values
#############################
BAYGpred.test <- predict(BAYGmodel, newdata = data.frame(testBin))

##################################################################################
#### 21. Xgboost model using looped CV
##################################################################################
#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
require(xgboost)
set.seed(56)

feat_remove <- c("Id", "SalePrice")

param <- list("objective" = "reg:linear", "eval_metric" = "rmse",
              "eta" = 0.1, "max.depth" = 2)
bst.cv = xgb.cv(param = param, 
                data = as.matrix(trainTrainFold1[!(names(trainTrainFold1) %in% feat_remove)]), 
                label = trainTrainFold1$SalePrice, 
                nfold = 100, 
                nrounds = 500)

# # "eta" = 1, "max.depth" = 2, nfold = 5, nrounds = 6, test-rmse:38584.324219+1554.610185
# bst <- xgboost(data = as.matrix(trainTrainFold1[!(names(trainTrainFold1) %in% feat_remove)]), 
#                label = trainTrainFold1$SalePrice, 
#                max.depth = 2, 
#                eta = 1, 
#                nround = 6,
#                objective = "reg:linear")

# For loop to find the best parameters: nfolds?
# nrounds check up to 500
# max.depth 2, 3 ... 10
# eta, 0.1, 0.2, ... 1.0

nrounds_min <- 1
max_depth_min <- 2
eta_min <- 0.1
rmse_min <- 10000000000 # Some large number
# nfolds_min <- 2

#for (i in 2:50) { # for max.depth
#  for(k in seq(0, 1, by = 0.1)) { # for eta
#   for(j in 2:50) {
    param <- list("objective" = "reg:linear", "eval_metric" = "rmse",
                  "eta" = k, "max.depth" = i)
    bst.cv = xgb.cv(param = param, 
                    data = as.matrix(trainTrainFold1[!(names(trainTrainFold1) %in% feat_remove)]), 
                    label = trainTrainFold1$SalePrice, 
                    nfold = 10, # Preset
                    nrounds = 300) # Preset
    
    # Determine minimum rsme error
    min_error <- min(bst.cv$evaluation_log$test_rmse_mean)
    # Determine optimal nrounds_min, max_depth_min, eta_min, rmse_min
    if (min_error < min(rmse_min)) {
      nrounds_min <- c(nrounds_min, which(bst.cv$evaluation_log$test_rmse_mean == min_error))
      max_depth_min <- c(max_depth_min, i)
      eta_min <- c(eta_min, k)
      # nfolds_min <- j
      rmse_min <- c(rmse_min, min_error)
      print(min_error)
      print(i)
      print(k)
    }
#    }
#  }
#}

bst <- xgboost(data = as.matrix(trainTrainFold1[!(names(trainTrainFold1) %in% feat_remove)]), 
              label = trainTrainFold1$SalePrice, 
              max.depth = 2, 
              eta = 0.25, 
              nround = 500,
              objective = "reg:linear")    

XGBOOSTpred.trainTrainFold1 <- predict(bst, as.matrix(trainTrainFold1[!(names(trainTrainFold1) %in% feat_remove)]))
XGBOOSTpred.trainTestFold1 <- predict(bst, as.matrix(trainTestFold1[!(names(trainTestFold1) %in% feat_remove)]))

myRMSLE(actual = trainTrainFold1$SalePrice, predicted = XGBOOSTpred.trainTrainFold1)
# 0.05089059
myRMSLE(actual = trainTestFold1$SalePrice, predicted = XGBOOSTpred.trainTestFold1)
# 0.1232121

#############################
# Predict actual test values
#############################
XGBOOSTpred.test <- predict(bst, newdata = as.matrix(testBin[!(names(testBin) %in% feat_remove)]))

# Id <- testBin$Id
# SalePrice <- XGBOOSTpred.test
# output <- data.frame(Id, SalePrice)
# write.csv(output, file = "housingXGBOOST.csv", row.names = FALSE)

##################################################################################
#### Stacking Using model: LM
#### Stacked models: RF, GBM,  XGBL, RIDGE, LASSO, ENET, SVML2
##################################################################################

set.seed(56)

# Add training results to full training data
# Full dataset is trainRandom
modelPred <- c(XGBOOSTpred.trainTestFold1, 
               GBMpred.trainTestFold2,
               LMpred.trainTestFold3,
               XGBTpred.trainTestFold4,
               XGBLpred.trainTestFold5,
               SVML2pred.trainTestFold6,
               RIDGEpred.trainTestFold7,
               LASSOpred.trainTestFold8,
               ENETpred.trainTestFold9,
               BAYGpred.trainTestFold10
               )

trainFinal <- data.frame(trainRandom, prediction = modelPred)
# Set up trainControl for CV
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)

# Run RF model
feat_remove <- c("Id")
RFmodel <- train(SalePrice ~ .,
                 method = "rf",
                 trControl = ctrl,
                 preProc = c("center", "scale"),
                 data = trainFinal[!(names(trainFinal) %in% feat_remove)]
)

RFpred.trainFinal <- predict(RFmodel, newdata = data.frame(trainFinal))
myRMSLE(actual = trainFinal$SalePrice, predicted = RFpred.trainFinal)

testPred <- (XGBOOSTpred.test
             + GBMpred.test
             + LMpred.test
             + XGBTpred.test
             + XGBLpred.test
             + SVML2pred.test
             + RIDGEpred.test
             + LASSOpred.test
             + ENETpred.test
             + BAYGpred.test)/10

testFinal <- data.frame(testBin, prediction = testPred)
RFTestpred.test <- predict(RFmodel, newdata = data.frame(testFinal))

#Id <- testBin$Id
#SalePrice <- RFTestpred.test
#output <- data.frame(Id, SalePrice)
#write.csv(output, file = "housingRFStacked.csv", row.names = FALSE)

testPredRF <- (XGBOOSTpred.test
             + GBMpred.test
             + LMpred.test
             + XGBTpred.test
             #+ XGBLpred.test
             + SVML2pred.test
             #+ RIDGEpred.test
             #+ LASSOpred.test
             + ENETpred.test
             #+ BAYGpred.test
             )/6

testFinalRF <- data.frame(testBin, prediction = testPredRF)
RFTestpred2.test <- predict(RFmodel, newdata = data.frame(testFinalRF))

Id <- testFinalRF$Id
SalePrice <- RFTestpred2.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingRFStacked2.csv", row.names = FALSE)

###############################################################################

set.seed(56)
ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
feat_remove <- c("Id")
XGBTFinalmodel <- train(SalePrice ~ .,
                   method = "xgbTree",
                   trControl = ctrl,
                   preProc = c("center", "scale"),
                   data = trainFinal[!(names(trainFinal) %in% feat_remove)]
)

testXGBTFinal <- data.frame(testBin, prediction = testPred)
XGBTTestpred.test <- predict(XGBTFinalmodel, newdata = data.frame(testXGBTFinal))

Id <- testFinal$Id
SalePrice <- XGBTTestpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingXGBTStacked.csv", row.names = FALSE)

###############################################################################

testALLPred <- (XGBOOSTpred.test
             + GBMpred.test
             + LMpred.test
             + XGBTpred.test
             + XGBLpred.test
             + SVML2pred.test
             + RIDGEpred.test
             + LASSOpred.test
             + ENETpred.test
             + BAYGpred.test)/10

Id <- testBin$Id
SalePrice <- testALLPred
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingALLmean.csv", row.names = FALSE)

###############################################################################

testALLlogPred <- exp((log(XGBOOSTpred.test)
                  + log(GBMpred.test)
                  + log(LMpred.test)
                  + log(XGBTpred.test)
                  + log(XGBLpred.test)
                  + log(SVML2pred.test)
                  + log(RIDGEpred.test)
                  + log(LASSOpred.test)
                  + log(ENETpred.test)
                  + log(BAYGpred.test))/10)

Id <- testBin$Id
SalePrice <- testALLlogPred
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingALLlogmean.csv", row.names = FALSE)

###############################################################################
# trainTrainSVML2_S <- data.frame(trainTrain, 
#                                 RFpred = RFpred.trainTrain,
#                                 GBMpred = GBMpred.trainTrain,
#                                 LMpred = LMpred.trainTrain,
#                                 XGBLpred = XGBLpred.trainTrain,
#                                 RIDGEpred = RIDGEpred.trainTrain,
#                                 LASSOpred = LASSOpred.trainTrain,
#                                 ENETpred = ENETpred.trainTrain,
#                                 POLYpred = POLYpred.trainTrain
# )
# 
# trainTestSVML2_S <- data.frame(trainTest, 
#                                 RFpred = RFpred.trainTest,
#                                 GBMpred = GBMpred.trainTest,
#                                 LMpred = LMpred.trainTest,
#                                 XGBLpred = XGBLpred.trainTest,
#                                 RIDGEpred = RIDGEpred.trainTest,
#                                 LASSOpred = LASSOpred.trainTest,
#                                 ENETpred = ENETpred.trainTest,
#                                 POLYpred = POLYpred.trainTest
# )
# 
# # Run model
# SVML2_S_model <- train(SalePrice ~ .,
#                        method = "svmLinear3",
#                        trControl = ctrl,
#                        preProc = c("center", "scale"),
#                        data = trainTrainSVML2_S[,-1]
# )
# 
# SVML2_S_pred.trainTrain <- predict(SVML2_S_model, newdata = data.frame(trainTrainSVML2_S))
# SVML2_S_pred.trainTest <- predict(SVML2_S_model, newdata = data.frame(trainTestSVML2_S))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = SVML2_S_pred.trainTrain)
# # 0.0238194
# myRMSLE(actual = trainTest$SalePrice, predicted = SVML2_S_pred.trainTest)
# # 0.1182356



# trainTrainLM_S <- data.frame(trainTrain, 
#                                 RFpred = RFpred.trainTrain,
#                                 GBMpred = GBMpred.trainTrain,
#                                 SVML2pred = SVML2pred.trainTrain,
#                                 XGBLpred = XGBLpred.trainTrain,
#                                 RIDGEpred = RIDGEpred.trainTrain,
#                                 LASSOpred = LASSOpred.trainTrain,
#                                 ENETpred = ENETpred.trainTrain,
#                                 POLYpred = POLYpred.trainTrain
# )
# 
# trainTestLM_S <- data.frame(trainTest, 
#                                RFpred = RFpred.trainTest,
#                                GBMpred = GBMpred.trainTest,
#                                SVML2pred = SVML2pred.trainTest,
#                                XGBLpred = XGBLpred.trainTest,
#                                RIDGEpred = RIDGEpred.trainTest,
#                                LASSOpred = LASSOpred.trainTest,
#                                ENETpred = ENETpred.trainTest,
#                                POLYpred = POLYpred.trainTest
# )
# 
# # Run model
# feat_remove <- c("Id", "MSSubClass.150", "Condition2.PosA", "MiscFeature.TenC", "PoolQC.Fa")
# LM_S_model <- train(SalePrice ~ .,
#                        method = "lm",
#                        trControl = ctrl,
#                        preProc = c("center", "scale"),
#                        data = trainTrainLM_S[!(names(trainTrain) %in% feat_remove)]
# )
# 
# LM_S_pred.trainTrain <- predict(LM_S_model, newdata = data.frame(trainTrainLM_S))
# LM_S_pred.trainTest <- predict(LM_S_model, newdata = data.frame(trainTestLM_S))
# 
# myRMSLE(actual = trainTrain$SalePrice, predicted = LM_S_pred.trainTrain)
# # 0.0238194
# myRMSLE(actual = trainTest$SalePrice, predicted = LM_S_pred.trainTest)
# # 0.1182356
