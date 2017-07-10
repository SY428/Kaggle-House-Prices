### House Prices: Advanced Regression Techniques
# Set workdir
setwd("/Users/sophieyang/Documents/Kaggle/HousePrices")

# Set seed
set.seed(428)

# Load Libraries
library(randomForest)
library(caret)
library(neuralnet)
library(MASS)
library(Metrics)
library(leaps)
library(glmnet)

# Get files
train <- read.csv("trainClean.csv")
test <- read.csv("testClean.csv")
test <- test[names(test) != "is_train"]
dataFinal <- read.csv("traintestClean.csv")
dataNoId <- dataFinal[names(dataFinal) != "Id"]


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

#####################################################################
### PREDICTIONS 1
#####################################################################
# Split up training set into training and testing sets
# First shuffle the data
# Split is around 75% training, 25% testing,
set.seed(234)

train1 <- train[sample(nrow(train)),]
inTrain <- createDataPartition(y = train1$SalePrice, p = 0.8, list = FALSE)
trainTrain <- train1[inTrain,]
trainTest <- train1[-inTrain,]

all_levels <- TRUE
k <- 0
for (i in 1:length(trainTrain)) {
  if(is.factor(trainTrain[,i])) {
    if (!all(unique(train1[,i]) %in% unique(trainTrain[,i]))) {
      all_levels <- FALSE
      k <- i
      break
    }
  }
}
iterations <- 1
while (!all_levels) {
  iterations <- iterations + 1
  set.seed(iterations)
  train1 <- train[sample(nrow(train)),]
  inTrain <- createDataPartition(y = train1$SalePrice, p = 0.75, list = FALSE)
  trainTrain <- train1[inTrain,]
  trainTest <- train1[-inTrain,]
  
  all_levels <- TRUE
  for (i in 1:length(trainTrain)) {
    if(is.factor(trainTrain[,i])) {
      if (!all(unique(train1[,i]) %in% unique(trainTrain[,i]))) {
        all_levels <- FALSE
        break
      }
    }
  }
}

dim(trainTrain)
dim(trainTest)

fitModel1 <- lm(SalePrice ~ ., data = trainTrain)
my_anova1 <- anova(fitModel1)
fitModel1_features <- row.names(subset(my_anova1, my_anova1$`Pr(>F)` < 0.01))

Model1 <- lm(SalePrice ~ . + YearBuilt*YearRemodAdd + YearBuilt*GarageArea + BsmtFinSF1*BsmtUnfSF + X1stFlrSF*BsmtFinSF1, data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "MSSubClass") & (names(trainTrain) != "PoolQC") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel1_features))])
#Model1 <- lm(SalePrice ~ ., data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "MSSubClass") & (names(trainTrain) != "PoolQC") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel1_features))])

Model1_SalePrice <- predict(Model1, newdata = data.frame(trainTest[names(trainTest) != "SalePrice"]))

# Check RMSE
rmse(actual = trainTest$SalePrice, predicted = Model1_SalePrice)
# recent 23176.86
#22485.93

###### TEST 1
Model1_Test_SalePrice <- predict(Model1, newdata = data.frame(test[names(test) != "SalePrice"]))

Id <- test$Id
SalePrice <- Model1_Test_SalePrice
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housing.csv", row.names = FALSE)
######

###########################################################################
## Check Correlation
###########################################################################

### Numeric predictors
colNum <- sapply(dataFinal, is.numeric)
colInt <- sapply(dataFinal, is.integer)

dataFullNum <- dataFinal[,colNum]
dataFullNum <- dataFinal[names(dataFinal) != "Id" & names(dataFinal) != "SalePrice"]
dataFullInt <- dataFinal[,colInt]

dataFullnonFactor <- dataFinal[,colNum | colInt]

corrNum <- cor(dataFullNum, dataFullNum)

corrThresh <- 0.2

pred1 <- as.character()
pred2 <- as.character()
corrCoef <- as.numeric()

for (i in 1:nrow(corrNum)) {
  len <- length(corrNum[i:nrow(corrNum),i][(corrNum[i:nrow(corrNum),i]>corrThresh | corrNum[i:nrow(corrNum),i]< -corrThresh) & (corrNum[i:nrow(corrNum),i] < max(corrNum[i:nrow(corrNum),i]))])
  if(len > 0) {
    for(k in 1:len) {
      pred1 <- c(pred1, names(corrNum[i:nrow(corrNum),i][corrNum[i:nrow(corrNum),i] == max(corrNum[i:nrow(corrNum),i])]))
      pred2 <- c(pred2, names(corrNum[i:nrow(corrNum),i][(corrNum[i:nrow(corrNum),i]>corrThresh | corrNum[i:nrow(corrNum),i]< -corrThresh) & (corrNum[i:nrow(corrNum),i] < max(corrNum[i:nrow(corrNum),i]))])[k])
      corrCoef <- c(corrCoef, corrNum[i:nrow(corrNum),i][(corrNum[i:nrow(corrNum),i]>corrThresh | corrNum[i:nrow(corrNum),i]< -corrThresh) & (corrNum[i:nrow(corrNum),i] < max(corrNum[i:nrow(corrNum),i]))][k])
    }
  }
}

corrNum.df <- data.frame(pred1 = pred1, pred2 = pred2, corrCoef = corrCoef)
corrNum.df <- corrNum.df[order(abs(corrNum.df$corrCoef),decreasing = TRUE),]

### Factors Predictors
colFac <- sapply(dataFinal, is.factor)

dataFullFactor <- dataFinal[,colFac]

### Chi Square Test p-values
dataFull_mod <- dataFull[names(dataFullFactor) != "Id" & names(dataFullFactor) != "is_train"]

cPred1 <- as.character()
cPred2 <- as.character()
cCoef <- as.numeric()

for (i in 1:length(dataFull_mod)) {
  #if (dataFull_mod[i:nrow(dataFull_mod),i]) {
  for (k in i:length(dataFull_mod)) {
    if(i < k) {
      chisqTest <- chisq.test(dataFull_mod[,i], dataFull_mod[,k])
      #if(chisqTest$p.value <= 0.001) {
      cPred1 <- c(cPred1, names(dataFull_mod)[i])
      cPred2 <- c(cPred2, names(dataFull_mod)[k])
      cCoef <- c(cCoef, chisqTest$p.value)
      #}
    }
  }
  #}
}

chisq.df <- data.frame(cPred1 = cPred1, cPred2 = cPred2, cCoef = cCoef)
chisq.df <- chisq.df[order(chisq.df$cCoef,decreasing = FALSE),]

###########################################################################
### PREDICTION 2
###########################################################################
set.seed(345)
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
# NEW PREDICTOR: TotFlrSF <- X1stFlrSF + X2ndFlrSF
TotFlrSF <- trainExp$X1stFlrSF + trainExp$X2ndFlrSF
trainExp <- data.frame(trainExp, TotFlrSF)
# Remove outliers for TotFlrSF
# plot -> plot(y=trainExp$SalePrice, x=trainExp$TotFlrSF)
trainExp <- droplevels(subset(trainExp, trainExp$TotFlrSF != max(trainExp$TotFlrSF)))
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
# Remove points where Utilities is "NoSeWa"
trainExp <- droplevels(subset(trainExp, trainExp$Utilities == "AllPub"))
# Remove Utilities
trainExp <- trainExp[names(trainExp) != "Utilities"]

### set as factors
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
# NEW PREDICTOR: TotFlrSF <- X1stFlrSF + X2ndFlrSF
TotFlrSF <- testExp$X1stFlrSF + testExp$X2ndFlrSF
testExp <- data.frame(testExp, TotFlrSF)
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
### set as factors
testExp$remodeled <- as.factor(testExp$remodeled)
testExp$Bangalow <- as.factor(testExp$Bangalow)
testExp$Wooddeck <- as.factor(testExp$Wooddeck)
testExp$Pool <- as.factor(testExp$Pool)
testExp$Bsmt <- as.factor(testExp$Bsmt)

# Implement PCA on training data
# Split up training set into training and testing sets
# First shuffle the data
# Split is around 75% training, 25% testing,

train2 <- trainExp[sample(nrow(trainExp)),]
inTrain <- createDataPartition(y = train2$SalePrice, p = 0.75, list = FALSE)
trainTrain <- train2[inTrain,]
trainTest <- train2[-inTrain,]

all_levels <- TRUE
k <- 0
for (i in 1:length(trainTrain)) {
  if(is.factor(trainTrain[,i])) {
    if (!all(unique(train2[,i]) %in% unique(trainTrain[,i]))) {
      all_levels <- FALSE
      k <- i
      break
    }
  }
}
iterations <- 1
while (!all_levels) {
  iterations <- iterations + 1
  set.seed(iterations)
  train2 <- trainExp[sample(nrow(trainExp)),]
  inTrain <- createDataPartition(y = train2$SalePrice, p = 0.75, list = FALSE)
  trainTrain <- train2[inTrain,]
  trainTest <- train2[-inTrain,]
  
  all_levels <- TRUE
  for (i in 1:length(trainTrain)) {
    if(is.factor(trainTrain[,i])) {
      if (!all(unique(train2[,i]) %in% unique(trainTrain[,i]))) {
        all_levels <- FALSE
        break
      }
    }
  }
}

dim(trainTrain)
dim(trainTest)

preProcPCA <- preProcess(trainTrain[(names(trainTrain) != "SalePrice") & (names(trainTrain) != "Id")], method = "pca", thresh = 0.99)
trainTrainPCA <- predict(preProcPCA, trainTrain[(names(trainTrain) != "SalePrice") & (names(trainTrain) != "Id")])
trainTrainPCA <- data.frame(SalePrice = trainTrain$SalePrice, trainTrainPCA)

fitModel2 <- lm(SalePrice ~ ., data = trainTrainPCA[(names(trainTrainPCA) != "Id") & (names(trainTrainPCA) != "MSSubClass")])
my_anova2 <- anova(fitModel2)
fitModel2_features <- row.names(subset(my_anova2, my_anova2$`Pr(>F)` < 0.0001))

Model2 <- lm(SalePrice ~ ., data = trainTrainPCA[(names(trainTrainPCA) != "Id") & ((names(trainTrainPCA) == "SalePrice") | (names(trainTrainPCA) %in% fitModel2_features))])

trainTestPCA <- predict(preProcPCA, trainTest[(names(trainTest) != "SalePrice") & (names(trainTest) != "Id")])
trainTestPCA <- data.frame(SalePrice = trainTest$SalePrice, trainTestPCA)

Model2_trainTrainPCA_SalePrice <- predict(Model2, newdata = data.frame(trainTrainPCA[names(trainTrainPCA) != "SalePrice"]))
Model2_trainTestPCA_SalePrice <- predict(Model2, newdata = data.frame(trainTestPCA[names(trainTestPCA) != "SalePrice"]))

# Check RMSE
rmse(actual = trainTrainPCA$SalePrice, predicted = Model2_trainTrainPCA_SalePrice)
rmse(actual = trainTestPCA$SalePrice, predicted = Model2_trainTestPCA_SalePrice)

########### Write to document
test2 <- testExp

testPCA <- predict(preProcPCA, test2)
Model2_testPCA_SalePrice <- predict(Model2, newdata = data.frame(testPCA))

Id <- test2$Id
SalePrice <- Model2_Test_SalePrice
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingPCA.csv", row.names = FALSE)

#####################################################################
### PREDICTIONS 3
#####################################################################
# Split up training set into training and testing sets
# First shuffle the data
# Split is around 80% training, 20% testing,
set.seed(428)

train3 <- train[sample(nrow(train)),]
inTrain <- createDataPartition(y = train3$SalePrice, p = 0.75, list = FALSE)
trainTrain <- train3[inTrain,]
trainTest <- train3[-inTrain,]

all_levels <- TRUE
k <- 0
for (i in 1:length(trainTrain)) {
  if(is.factor(trainTrain[,i])) {
    if (!all(unique(train3[,i]) %in% unique(trainTrain[,i]))) {
      all_levels <- FALSE
      k <- i
      break
    }
  }
}
iterations <- 1
while (!all_levels) {
  iterations <- iterations + 1
  set.seed(iterations)
  train3 <- train[sample(nrow(train)),]
  inTrain <- createDataPartition(y = train3$SalePrice, p = 0.75, list = FALSE)
  trainTrain <- train3[inTrain,]
  trainTest <- train3[-inTrain,]
  
  all_levels <- TRUE
  for (i in 1:length(trainTrain)) {
    if(is.factor(trainTrain[,i])) {
      if (!all(unique(train3[,i]) %in% unique(trainTrain[,i]))) {
        all_levels <- FALSE
        break
      }
    }
  }
}

dim(trainTrain)
dim(trainTest)

# Original
fitModel3 <- lm(SalePrice ~ ., data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "MSSubClass")])
my_anova3 <- anova(fitModel3)
fitModel3_features <- row.names(subset(my_anova3, my_anova3$`Pr(>F)` < 0.001))

# Model3 <- lm(SalePrice ~ . + YearBuilt*YearRemodAdd + YearBuilt*GarageArea + BsmtFinSF1*BsmtUnfSF + X1stFlrSF*BsmtFinSF1, data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel3_features))])
# 23393.71, features < 0.1, train = 0.75
# 22323.58, features < 0.01, train = 0.75

# Model3 <- lm(SalePrice ~ . + YearBuilt*YearRemodAdd + BsmtFinSF1*BsmtUnfSF + X1stFlrSF*BsmtFinSF1, data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel3_features))])
# 22259.9, features < 0.01, train = 0.75
# 21960.32, features < 0.001, train = 0.75
# 21812.49, features < 0.1, train = 0.751

 Model3 <- lm(SalePrice ~ . + X1stFlrSF*BsmtFinSF1, data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel3_features))])
# 21779.91, features < 0.001, train = 0.75

Model3_trainTrain_SalePrice <- predict(Model3, newdata = data.frame(trainTrain[names(trainTrain) != "SalePrice"]))
Model3_trainTest_SalePrice <- predict(Model3, newdata = data.frame(trainTest[names(trainTest) != "SalePrice"]))

# Check RMSE
rmse(actual = trainTrain$SalePrice, predicted = Model3_trainTrain_SalePrice)
rmse(actual = trainTest$SalePrice, predicted = Model3_trainTest_SalePrice)


# New pool

fitModelNA3 <- lm(SalePrice ~ ., data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "MSSubClass") & (names(trainTrain) != "PoolQC")])
my_anovaNA3 <- anova(fitModelNA3)
fitModelNA3_features <- row.names(subset(my_anovaNA3, my_anovaNA3$`Pr(>F)` < 0.01))
ModelNA3 <- lm(SalePrice ~ . + YearBuilt*YearRemodAdd + BsmtFinSF1*BsmtUnfSF + X1stFlrSF*BsmtFinSF1, data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModelNA3_features))])

###### TEST 3 FULL
test3_full <- test
test3_full_pool <- test3_full[which(test3_full$PoolQC != "Fa"),]
test3_full_NApool <- test3_full[which(test3_full$PoolQC == "Fa"),]

Model3_Test_SalePrice_NApool <- predict(ModelNA3, newdata = data.frame(test3_full_NApool[names(test3_full_NApool) != "SalePrice"]))
Model3_Test_SalePrice_nonNApool <- predict(Model3, newdata = data.frame(test3_full_pool[names(test3_full_pool) != "SalePrice"]))

Id <- c(test3_full_pool$Id, test3_full_NApool$Id)
SalePrice <- c(Model3_Test_SalePrice_nonNApool, Model3_Test_SalePrice_NApool)
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housing.csv", row.names = FALSE)
######

# Splitting into having pool and not having pool
trainTrain_nopool <- subset(trainTrain, trainTrain$PoolQC == "NoPool")
trainTrain_pool <- subset(trainTrain, trainTrain$PoolQC != "NoPool")
trainTest_nopool <- subset(trainTest, trainTest$PoolQC == "NoPool")
trainTest_pool <- subset(trainTest, trainTest$PoolQC != "NoPool")

fitModel3_nopool <- lm(SalePrice ~ ., data = trainTrain_nopool[(names(trainTrain_nopool) != "Id") & (names(trainTrain_nopool) != "PoolQC")])
my_anova3_nopool <- anova(fitModel3_nopool)
fitModel3_nopool_features <- row.names(subset(my_anova3_nopool, my_anova3_nopool$`Pr(>F)` < 0.01))
#Model3_nopool <- lm(SalePrice ~ . + YearBuilt*YearRemodAdd + YearBuilt*GarageArea + BsmtFinSF1*BsmtUnfSF + X1stFlrSF*BsmtFinSF1, data = trainTrain_nopool[(names(trainTrain_nopool) != "Id") & (names(trainTrain_nopool) != "PoolQC") & ((names(trainTrain_nopool) == "SalePrice") | (names(trainTrain_nopool) %in% fitModel3_nopool_features))])
Model3_nopool <- lm(SalePrice ~ . + YearBuilt*YearRemodAdd + BsmtFinSF1*BsmtUnfSF + X1stFlrSF*BsmtFinSF1, data = trainTrain_nopool[(names(trainTrain_nopool) != "Id") & (names(trainTrain_nopool) != "PoolQC") & ((names(trainTrain_nopool) == "SalePrice") | (names(trainTrain_nopool) %in% fitModel3_nopool_features))])
Model3_trainTrain_nopool_SalePrice <- predict(Model3_nopool, newdata = data.frame(trainTrain_nopool[names(trainTrain_nopool) != "SalePrice"]))
Model3_trainTest_nopool_SalePrice <- predict(Model3_nopool, newdata = data.frame(trainTest_nopool[names(trainTest_nopool) != "SalePrice"]))

names_features <- c("Street", "LandSlope", "Condition2", "Heating", "CentralAir", "PavedDrive", "Electrical", "Utilities", "Functional", "MiscFeature", "Id")
fitModel3_pool <- lm(SalePrice ~ ., data = trainTrain_pool[!(names(trainTrain_pool) %in% names_features)])
my_anova3_pool <- anova(fitModel3_pool)
#fitModel3_pool_features <- row.names(subset(my_anova3_pool, my_anova3_pool$`Pr(>F)` < 0.01))
fitModel3_pool_features <- row.names(subset(my_anova3_pool, my_anova3_pool$`Mean Sq` > 0))
Model3_pool <- lm(SalePrice ~ ., data = trainTrain_pool[(names(trainTrain_pool) != "Id") & ((names(trainTrain_pool) == "SalePrice") | (names(trainTrain_pool) %in% fitModel3_pool_features))])
Model3_trainTrain_pool_SalePrice <- predict(Model3_pool, newdata = data.frame(trainTrain_pool[names(trainTrain_pool) != "SalePrice"]))
Model3_trainTest_pool_SalePrice <- predict(Model3_pool, newdata = data.frame(trainTest_pool[names(trainTest_pool) != "SalePrice"]))

# Check RMSE
rmse(actual = rbind(trainTrain_nopool$SalePrice, trainTrain_pool$SalePrice), predicted = rbind(Model3_trainTrain_nopool_SalePrice, Model3_trainTrain_pool_SalePrice))
rmse(actual = rbind(trainTest_nopool$SalePrice, trainTest_pool$SalePrice), predicted = rbind(Model3_trainTest_nopool_SalePrice, Model3_trainTest_pool_SalePrice))

#set.seed(428)
#p = 0.75, 22602.76
#p = 0.76, 21359.72

###### TEST 3 part 2
test3 <- test

test3_nopool <- subset(test3, test3$PoolQC == "NoPool")
test3_pool <- subset(test3, test3$PoolQC != "NoPool")

Model3_Test_nopool_SalePrice <- predict(Model3, newdata = data.frame(test3[names(test3) != "SalePrice"]))
Model3_Test_pool_SalePrice <- predict(Model3, newdata = data.frame(test3[names(test3) != "SalePrice"]))

Id <- test3$Id
SalePrice <- Model3_Test_SalePrice
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housing.csv", row.names = FALSE)
######

#####################################################################
### PREDICTIONS 4
#####################################################################
# Split up training set into training and testing sets
# First shuffle the data
# Split is around 75% training, 25% testing,
set.seed(428)

train4 <- train[sample(nrow(train)),]
inTrain <- createDataPartition(y = train4$SalePrice, p = 0.75, list = FALSE)
trainTrain <- train4[inTrain,]
trainTest <- train4[-inTrain,]

all_levels <- TRUE
k <- 0
for (i in 1:length(trainTrain)) {
  if(is.factor(trainTrain[,i])) {
    if (!all(unique(train4[,i]) %in% unique(trainTrain[,i]))) {
      all_levels <- FALSE
      k <- i
      break
    }
  }
}
iterations <- 1
while (!all_levels) {
  iterations <- iterations + 1
  set.seed(iterations)
  train4 <- train[sample(nrow(train)),]
  inTrain <- createDataPartition(y = train4$SalePrice, p = 0.75, list = FALSE)
  trainTrain <- train4[inTrain,]
  trainTest <- train4[-inTrain,]
  
  all_levels <- TRUE
  for (i in 1:length(trainTrain)) {
    if(is.factor(trainTrain[,i])) {
      if (!all(unique(train4[,i]) %in% unique(trainTrain[,i]))) {
        all_levels <- FALSE
        break
      }
    }
  }
}

dim(trainTrain)
dim(trainTest)


fitModel4 <- lm(SalePrice ~ ., data = trainTrain[(names(trainTrain) != "Id")])
my_anova4 <- anova(fitModel4)
fitModel4_features <- row.names(subset(my_anova4, my_anova4$`Pr(>F)` < 0.01))
all_features <- row.names(my_anova4)

#regfit.full <- regsubsets(SalePrice~., data = trainTrain[(names(trainTrain) != "Id")], method = c("forward"))
#regfit.full <- regsubsets(SalePrice~., data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel4_features))], method = c("forward"), nvmax = length(fitModel4_features))
regfit.full <- regsubsets(SalePrice~., data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% all_features))], method = c("forward"), nvmax = length(all_features)-2)
reg.summary <- summary(regfit.full)
plot(reg.summary$cp)
which.min(reg.summary$cp)

#####################################################################
### PREDICTIONS 5 Ridge
#####################################################################
# Split up training set into training and testing sets
# First shuffle the data
# Split is around 75% training, 25% testing,
set.seed(428)

train5 <- trainExp[sample(nrow(trainExp)),]
inTrain <- createDataPartition(y = train5$SalePrice, p = 0.75, list = FALSE)
trainTrain <- train5[inTrain,]
trainTest <- train5[-inTrain,]

all_levels <- TRUE
k <- 0
for (i in 1:length(trainTrain)) {
  if(is.factor(trainTrain[,i])) {
    if (!all(unique(train5[,i]) %in% unique(trainTrain[,i]))) {
      all_levels <- FALSE
      k <- i
      break
    }
  }
}
iterations <- 1
while (!all_levels) {
  iterations <- iterations + 1
  set.seed(iterations)
  train5 <- trainExp[sample(nrow(trainExp)),]
  inTrain <- createDataPartition(y = train5$SalePrice, p = 0.75, list = FALSE)
  trainTrain <- train5[inTrain,]
  trainTest <- train5[-inTrain,]
  
  all_levels <- TRUE
  for (i in 1:length(trainTrain)) {
    if(is.factor(trainTrain[,i])) {
      if (!all(unique(train5[,i]) %in% unique(trainTrain[,i]))) {
        all_levels <- FALSE
        break
      }
    }
  }
}

dim(trainTrain)
dim(trainTest)

x <- model.matrix(SalePrice~., data = trainTrain[(names(trainTrain) != "Id")])
y <- trainTrain$SalePrice

fit.ridge <- glmnet(x, y, alpha = 0)
plot(fit.ridge, xvar = "lambda", label = TRUE)
cv.ridge <- cv.glmnet(x, y, alpha = 0)
plot(cv.ridge)

xtest <- model.matrix(SalePrice~., data = trainTest[(names(trainTest) != "Id")])
ytest <- trainTest$SalePrice

ridge.tr <- glmnet(x, y, alpha = 0, nlambda = 1000)
pred <- predict(ridge.tr, xtest)
rmse <- sqrt(apply((ytest-pred)^2, 2, mean))
plot(log(ridge.tr$lambda), rmse, type = "b", xlab = "log(lambda)")
lam.best <- ridge.tr$lambda[which(rmse == min(rmse))]
rmse[which(rmse == min(rmse))]
lam.best
Others_sel <- which(ridge.tr$lambda==lam.best)

# Predicting actual test
test <- droplevels(test)
xFtest <- model.matrix(SalePrice~., data = test[(names(test) != "Id")])

# Before predicting, create dummy tests to fill factors
feature_lvls_A <- names(x[1,])[!(names(x[1,]) %in% names(xFtest[1,]))]
feature_lvls_B <- names(xFtest[1,])[!(names(xFtest[1,]) %in% names(x[1,]))]

testridge <- test

pts_to_add <- train5[which(train5$Condition2 == "RRAe"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$Condition2 == "RRAn"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$Condition2 == "RRNn"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$HouseStyle == "2.5Fin"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$RoofMatl == "CompShg"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$RoofMatl == "Membran"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$RoofMatl == "Metal"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$RoofMatl == "Roll"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$RoofMatl == "ClyTile"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$Heating == "GasA"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$Heating == "OthW"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$Heating == "Floor"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$Exterior1st == "ImStucc"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$Exterior1st == "Stone"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$Exterior2nd == "Other"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$Electrical == "Mix"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$GarageQual == "Fa"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$GarageQual == "Ex"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$GarageCond == "Fa"),][1,]
testridge <- rbind(testridge, pts_to_add)
pts_to_add <- train5[which(train5$GarageCond == "Ex"),][1,]
testridge <- rbind(testridge, pts_to_add)

xF2test <- model.matrix(SalePrice~., data = testridge[(names(testridge) != "Id") & (names(testridge) != "Utilities")])

feature_lvls_C <- names(x[1,])[!(names(x[1,]) %in% names(xF2test[1,]))]
feature_lvls_D <- names(xF2test[1,])[!(names(xF2test[1,]) %in% names(x[1,]))]

testridgeOthers <- droplevels(subset(testridge,(testridge$PoolQC != "Fa") & (testridge$MSSubClass != "150") & (testridge$MiscFeature != "TenC")))
testridgeMiscFeature <- rbind(droplevels(subset(testridge,(testridge$MiscFeature == "TenC"))), testridgeOthers)
testridgeMSSubClass <- rbind(droplevels(subset(testridge,(testridge$MSSubClass == "150"))), testridgeOthers)
testridgePoolQC <- rbind(droplevels(subset(testridge,(testridge$PoolQC == "Fa"))), testridgeOthers)

xtestOthers <- model.matrix(SalePrice~., data = testridgeOthers[(names(testridgeOthers) != "Id") & (names(testridgeOthers) != "Utilities")])
predOthers <- predict(ridge.tr, xtestOthers)
predOthersBest <- predOthers[,Others_sel]

# Generating test for MiscFeature
x <- model.matrix(SalePrice~., data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "Utilities") & (names(trainTrain) != "MiscFeature")])
y <- trainTrain$SalePrice
fit.ridge <- glmnet(x, y, alpha = 0)
cv.ridge <- cv.glmnet(x, y, alpha = 0)
xtest <- model.matrix(SalePrice~., data = trainTest[(names(trainTest) != "Id") & (names(trainTest) != "Utilities") & (names(trainTest) != "MiscFeature")])
ytest <- trainTest$SalePrice
ridge.tr <- glmnet(x, y, alpha = 0)
pred <- predict(ridge.tr, xtest)
rmse <- sqrt(apply((ytest-pred)^2, 2, mean))
MiscFeaturelam.best <- ridge.tr$lambda[which(rmse == min(rmse))]
MiscFeature_sel <- which(ridge.tr$lambda==MiscFeaturelam.best)

xtestMiscFeature <- model.matrix(SalePrice~., data = testridgeMiscFeature[(names(testridgeMiscFeature) != "Id") & (names(testridgeMiscFeature) != "Utilities") & (names(testridgeMiscFeature) != "MiscFeature")])
predMiscFeature <- predict(ridge.tr, xtestMiscFeature)
predMiscFeatureTencBest <- predMiscFeature[1,MiscFeature_sel]

# Generating test for MSSubClass
x <- model.matrix(SalePrice~., data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "Utilities") & (names(trainTrain) != "MSSubClass")])
y <- trainTrain$SalePrice
fit.ridge <- glmnet(x, y, alpha = 0)
cv.ridge <- cv.glmnet(x, y, alpha = 0)
xtest <- model.matrix(SalePrice~., data = trainTest[(names(trainTest) != "Id") & (names(trainTest) != "Utilities") & (names(trainTest) != "MSSubClass")])
ytest <- trainTest$SalePrice
ridge.tr <- glmnet(x, y, alpha = 0)
pred <- predict(ridge.tr, xtest)
rmse <- sqrt(apply((ytest-pred)^2, 2, mean))
MSSubClasslam.best <- ridge.tr$lambda[which(rmse == min(rmse))]
MSSubClass_sel <- which(ridge.tr$lambda==MSSubClasslam.best)

xtestMSSubClass <- model.matrix(SalePrice~., data = testridgeMSSubClass[(names(testridgeMSSubClass) != "Id") & (names(testridgeMSSubClass) != "Utilities") & (names(testridgeMSSubClass) != "MSSubClass")])
predMSSubClass <- predict(ridge.tr, xtestMSSubClass)
predMSSubClass150Best <- predMSSubClass[1,MSSubClass_sel]

# Generating test for PoolQC
x <- model.matrix(SalePrice~., data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "Utilities") & (names(trainTrain) != "PoolQC")])
y <- trainTrain$SalePrice
fit.ridge <- glmnet(x, y, alpha = 0)
cv.ridge <- cv.glmnet(x, y, alpha = 0)
xtest <- model.matrix(SalePrice~., data = trainTest[(names(trainTest) != "Id") & (names(trainTest) != "Utilities") & (names(trainTest) != "PoolQC")])
ytest <- trainTest$SalePrice
ridge.tr <- glmnet(x, y, alpha = 0)
pred <- predict(ridge.tr, xtest)
rmse <- sqrt(apply((ytest-pred)^2, 2, mean))
PoolQClam.best <- ridge.tr$lambda[which(rmse == min(rmse))]
PoolQC_sel <- which(ridge.tr$lambda==PoolQClam.best)

xtestPoolQC <- model.matrix(SalePrice~., data = testridgePoolQC[(names(testridgePoolQC) != "Id") & (names(testridgePoolQC) != "Utilities") & (names(testridgePoolQC) != "PoolQC")])
predPoolQC <- predict(ridge.tr, xtestPoolQC)
predPoolQCFaBest <- predPoolQC[1:3,PoolQC_sel]

# Combine the predictions
head(predOthersBest)
predMiscFeatureTencBest
predMSSubClass150Best
predPoolQCFaBest

Id <- c(testridgeOthers$Id[1:1454], testridgeMiscFeature$Id[1], testridgeMSSubClass$Id[1], testridgePoolQC$Id[1:3])
SalePrice <- c(predOthersBest[1:1454], predMiscFeatureTencBest, predMSSubClass150Best, predPoolQCFaBest)
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingridge.csv", row.names = FALSE)

#####################################################################
### PREDICTIONS 6 LASSO
#####################################################################
# Split up training set into training and testing sets
# First shuffle the data
# Split is around 75% training, 25% testing,
set.seed(428)

train6 <- trainExp[sample(nrow(trainExp)),]
inTrain <- createDataPartition(y = train6$SalePrice, p = 0.75, list = FALSE)
trainTrain <- train6[inTrain,]
trainTest <- train6[-inTrain,]

all_levels <- TRUE
k <- 0
for (i in 1:length(trainTrain)) {
  if(is.factor(trainTrain[,i])) {
    if (!all(unique(train6[,i]) %in% unique(trainTrain[,i]))) {
      all_levels <- FALSE
      k <- i
      break
    }
  }
}
iterations <- 1
while (!all_levels) {
  iterations <- iterations + 1
  set.seed(iterations)
  train6 <- trainExp[sample(nrow(trainExp)),]
  inTrain <- createDataPartition(y = train6$SalePrice, p = 0.75, list = FALSE)
  trainTrain <- train6[inTrain,]
  trainTest <- train6[-inTrain,]
  
  all_levels <- TRUE
  for (i in 1:length(trainTrain)) {
    if(is.factor(trainTrain[,i])) {
      if (!all(unique(train6[,i]) %in% unique(trainTrain[,i]))) {
        all_levels <- FALSE
        break
      }
    }
  }
}

dim(trainTrain)
dim(trainTest)

x <- model.matrix(SalePrice~., data = trainTrain[(names(trainTrain) != "Id")])
y <- trainTrain$SalePrice

fit.lasso <- glmnet(x, y, alpha = 1, nlambda = 1000)
#plot(fit.lasso, xvar = "lambda", label = TRUE)
#plot(fit.lasso, xvar = "dev", label = TRUE)
cv.lasso <- cv.glmnet(x, y, alpha = 1)
#plot(cv.lasso)

xtest <- model.matrix(SalePrice~., data = trainTest[(names(trainTest) != "Id")])
ytest <- trainTest$SalePrice

lasso.tr <- glmnet(x, y, alpha = 1, nlambda = 1000)
pred <- predict(lasso.tr, xtest)
rmse <- sqrt(apply((ytest-pred)^2, 2, mean))
#plot(log(lasso.tr$lambda), rmse, type = "b", xlab = "log(lambda)")
lam.best <- lasso.tr$lambda[order(rmse)[1]]
rmse[order(rmse)[1]]
lam.best

Others_sel <- which(lasso.tr$lambda==lam.best)

testlasso <- test

pts_to_add <- train6[which(train6$Condition2 == "RRAe"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$Condition2 == "RRAn"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$Condition2 == "RRNn"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$HouseStyle == "2.5Fin"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$RoofMatl == "CompShg"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$RoofMatl == "Membran"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$RoofMatl == "Metal"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$RoofMatl == "Roll"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$RoofMatl == "ClyTile"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$Heating == "GasA"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$Heating == "OthW"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$Heating == "Floor"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$Exterior1st == "ImStucc"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$Exterior1st == "Stone"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$Exterior2nd == "Other"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$Electrical == "Mix"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$GarageQual == "Fa"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$GarageQual == "Ex"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$GarageCond == "Fa"),][1,]
testlasso <- rbind(testlasso, pts_to_add)
pts_to_add <- train6[which(train6$GarageCond == "Ex"),][1,]
testlasso <- rbind(testlasso, pts_to_add)

testlassoOthers <- droplevels(subset(testlasso,(testlasso$PoolQC != "Fa") & (testlasso$MSSubClass != "150") & (testlasso$MiscFeature != "TenC")))
testlassoMiscFeature <- rbind(droplevels(subset(testlasso,(testlasso$MiscFeature == "TenC"))), testlassoOthers)
testlassoMSSubClass <- rbind(droplevels(subset(testlasso,(testlasso$MSSubClass == "150"))), testlassoOthers)
testlassoPoolQC <- rbind(droplevels(subset(testlasso,(testlasso$PoolQC == "Fa"))), testlassoOthers)

xtestOthers <- model.matrix(SalePrice~., data = testlassoOthers[(names(testlassoOthers) != "Id") & (names(testlassoOthers) != "Utilities")])
predOthers <- predict(lasso.tr, xtestOthers)
predOthersBest <- predOthers[,Others_sel]

# Generating test for MiscFeature
x <- model.matrix(SalePrice~., data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "Utilities") & (names(trainTrain) != "MiscFeature")])
y <- trainTrain$SalePrice
fit.lasso <- glmnet(x, y, alpha = 1)
cv.lasso <- cv.glmnet(x, y, alpha = 1)
xtest <- model.matrix(SalePrice~., data = trainTest[(names(trainTest) != "Id") & (names(trainTest) != "Utilities") & (names(trainTest) != "MiscFeature")])
ytest <- trainTest$SalePrice
lasso.tr <- glmnet(x, y, alpha = 1)
pred <- predict(lasso.tr, xtest)
rmse <- sqrt(apply((ytest-pred)^2, 2, mean))
MiscFeaturelam.best <- lasso.tr$lambda[which(rmse == min(rmse))]
MiscFeature_sel <- which(lasso.tr$lambda==MiscFeaturelam.best)

xtestMiscFeature <- model.matrix(SalePrice~., data = testlassoMiscFeature[(names(testlassoMiscFeature) != "Id") & (names(testlassoMiscFeature) != "Utilities") & (names(testlassoMiscFeature) != "MiscFeature")])
predMiscFeature <- predict(lasso.tr, xtestMiscFeature)
predMiscFeatureTencBest <- predMiscFeature[1,MiscFeature_sel]

# Generating test for MSSubClass
x <- model.matrix(SalePrice~., data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "Utilities") & (names(trainTrain) != "MSSubClass")])
y <- trainTrain$SalePrice
fit.lasso <- glmnet(x, y, alpha = 1)
cv.lasso <- cv.glmnet(x, y, alpha = 1)
xtest <- model.matrix(SalePrice~., data = trainTest[(names(trainTest) != "Id") & (names(trainTest) != "Utilities") & (names(trainTest) != "MSSubClass")])
ytest <- trainTest$SalePrice
lasso.tr <- glmnet(x, y, alpha = 1)
pred <- predict(lasso.tr, xtest)
rmse <- sqrt(apply((ytest-pred)^2, 2, mean))
MSSubClasslam.best <- lasso.tr$lambda[which(rmse == min(rmse))]
MSSubClass_sel <- which(lasso.tr$lambda==MSSubClasslam.best)

xtestMSSubClass <- model.matrix(SalePrice~., data = testlassoMSSubClass[(names(testlassoMSSubClass) != "Id") & (names(testlassoMSSubClass) != "Utilities") & (names(testlassoMSSubClass) != "MSSubClass")])
predMSSubClass <- predict(lasso.tr, xtestMSSubClass)
predMSSubClass150Best <- predMSSubClass[1,MSSubClass_sel]

# Generating test for PoolQC
x <- model.matrix(SalePrice~., data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "Utilities") & (names(trainTrain) != "PoolQC")])
y <- trainTrain$SalePrice
fit.lasso <- glmnet(x, y, alpha = 1)
cv.lasso <- cv.glmnet(x, y, alpha = 1)
xtest <- model.matrix(SalePrice~., data = trainTest[(names(trainTest) != "Id") & (names(trainTest) != "Utilities") & (names(trainTest) != "PoolQC")])
ytest <- trainTest$SalePrice
lasso.tr <- glmnet(x, y, alpha = 1)
pred <- predict(lasso.tr, xtest)
rmse <- sqrt(apply((ytest-pred)^2, 2, mean))
PoolQClam.best <- lasso.tr$lambda[which(rmse == min(rmse))]
PoolQC_sel <- which(lasso.tr$lambda==PoolQClam.best)

xtestPoolQC <- model.matrix(SalePrice~., data = testlassoPoolQC[(names(testlassoPoolQC) != "Id") & (names(testlassoPoolQC) != "Utilities") & (names(testlassoPoolQC) != "PoolQC")])
predPoolQC <- predict(lasso.tr, xtestPoolQC)
predPoolQCFaBest <- predPoolQC[1:3,PoolQC_sel]

# Combine the predictions
head(predOthersBest)
predMiscFeatureTencBest
predMSSubClass150Best
predPoolQCFaBest

Id <- c(testlassoOthers$Id[1:1474], testlassoMiscFeature$Id[1], testlassoMSSubClass$Id[1], testlassoPoolQC$Id[1:3])
SalePrice <- c(predOthersBest[1:1474], predMiscFeatureTencBest, predMSSubClass150Best, predPoolQCFaBest)
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housinglasso.csv", row.names = FALSE)

#####################################################################
### PREDICTIONS 7 Grouping
#####################################################################
# Select different groups
# 1000 is the split
summary(train$SalePrice)

#####################################################################
### PREDICTIONS 8 regression with exploratory analysis
#####################################################################
set.seed(1)

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
# NEW PREDICTOR: TotFlrSF <- X1stFlrSF + X2ndFlrSF
TotFlrSF <- trainExp$X1stFlrSF + trainExp$X2ndFlrSF
trainExp <- data.frame(trainExp, TotFlrSF)
# Remove outliers for TotFlrSF
# plot -> plot(y=trainExp$SalePrice, x=trainExp$TotFlrSF)
trainExp <- droplevels(subset(trainExp, trainExp$TotFlrSF != max(trainExp$TotFlrSF)))
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
# Remove points where Utilities is "NoSeWa"
trainExp <- droplevels(subset(trainExp, trainExp$Utilities == "AllPub"))
# Remove Utilities
trainExp <- trainExp[names(trainExp) != "Utilities"]

### set as factors
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
# NEW PREDICTOR: TotFlrSF <- X1stFlrSF + X2ndFlrSF
TotFlrSF <- testExp$X1stFlrSF + testExp$X2ndFlrSF
testExp <- data.frame(testExp, TotFlrSF)
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
### set as factors
testExp$remodeled <- as.factor(testExp$remodeled)
testExp$Bangalow <- as.factor(testExp$Bangalow)
testExp$Wooddeck <- as.factor(testExp$Wooddeck)
testExp$Pool <- as.factor(testExp$Pool)
testExp$Bsmt <- as.factor(testExp$Bsmt)

# Split up training set into training and testing sets
# First shuffle the data
# Split is around 75% training, 25% testing
train8 <- trainExp[sample(nrow(trainExp)),]
inTrain <- createDataPartition(y = train8$SalePrice, p = 0.74, list = FALSE)
trainTrain <- train8[inTrain,]
trainTest <- train8[-inTrain,]

all_levels <- TRUE
k <- 0
for (i in 1:length(trainTrain)) {
  if(is.factor(trainTrain[,i])) {
    if (!all(unique(train8[,i]) %in% unique(trainTrain[,i]))) {
      all_levels <- FALSE
      k <- i
      break
    }
  }
}
iterations <- 1
while (!all_levels) {
  iterations <- iterations + 1
  set.seed(iterations)
  train8 <- trainExp[sample(nrow(trainExp)),]
  inTrain <- createDataPartition(y = train8$SalePrice, p = 0.74, list = FALSE)
  trainTrain <- train8[inTrain,]
  trainTest <- train8[-inTrain,]
  
  all_levels <- TRUE
  for (i in 1:length(trainTrain)) {
    if(is.factor(trainTrain[,i])) {
      if (!all(unique(train8[,i]) %in% unique(trainTrain[,i]))) {
        all_levels <- FALSE
        break
      }
    }
  }
}

dim(trainTrain)
dim(trainTest)

featRemoved <- c("Id", "MSSubClass", "Alley", "Fence", "FireplaceQu", "MasVnrType", "BsmtFinType1", "Exterior2nd", "GarageQual", "BsmtQual", "TotFlrSF", "PoolArea")
fitModel8 <- lm(SalePrice ~ ., data = trainTrain[!(names(trainTrain) %in% featRemoved)])
#fitModel8 <- lm(SalePrice ~ ., data = trainTrain[(names(trainTrain) != "Id") & (names(trainTrain) != "MSSubClass")])
my_anova8 <- anova(fitModel8)
fitModel8_features <- row.names(subset(my_anova8, my_anova8$`Pr(>F)` < 0.000001))

# Model8 <- lm(SalePrice ~ ., data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel8_features))])
  # 18374.35, features < 0.01, train = 0.75
# 18417.75, features < 0.001, train = 0.74
 
# Model8 <- lm(SalePrice ~ ., data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel8_features))])

# Model8 <- lm(SalePrice ~ . + YearBuilt*YearRemodAdd + YearBuilt*GarageArea + BsmtFinSF1*BsmtUnfSF + X1stFlrSF*BsmtFinSF1, data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel8_features))])
  # 18534.72, features < 0.01, train = 0.75

# Model8 <- lm(SalePrice ~ . + YearBuilt*YearRemodAdd + BsmtFinSF1*BsmtUnfSF + X1stFlrSF*BsmtFinSF1, data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel8_features))])
  # 18384.47, features < 0.001, train = 0.75
  # 18113.12, features < 0.0001, train = 0.75
# 18344.08, features < 0.0001, train = 0.74

 Model8 <- lm(SalePrice ~ . + X1stFlrSF*BsmtFinSF1, data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% fitModel8_features))])
  # 18321.42, features < 0.001, train = 0.75
  # 18077.68, features < 0.0001, train = 0.75
  # 17627.68, features < 0.00001, train = 0.75
# 18642.51, features < 0.0001, train = 0.74
# 18574.61, features < 0.00001, train = 0.74
# 18292.43, features < 0.000001, train = 0.74
 
Model8_trainTrain_SalePrice <- predict(Model8, newdata = data.frame(trainTrain[names(trainTrain) != "SalePrice"]))
Model8_trainTest_SalePrice <- predict(Model8, newdata = data.frame(trainTest[names(trainTest) != "SalePrice"]))

# Check RMSE
rmse(actual = trainTrain$SalePrice, predicted = Model8_trainTrain_SalePrice)
rmse(actual = trainTest$SalePrice, predicted = Model8_trainTest_SalePrice)

myRMSLE(actual = trainTrain$SalePrice, predicted = Model8_trainTrain_SalePrice)
myRMSLE(actual = trainTest$SalePrice, predicted = Model8_trainTest_SalePrice)

test8 <- testExp

Model8_Test_SalePrice <- predict(Model8, newdata = data.frame(test8[names(test8) != "SalePrice"]))

Id <- test8$Id
SalePrice <- Model8_Test_SalePrice
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingFeatureExt1.csv", row.names = FALSE)

#####################################################################
### PREDICTIONS 9 Select by hand features
#####################################################################
# Use the same feature extration as prediction 8
set.seed(1)

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
# NEW PREDICTOR: TotFlrSF <- X1stFlrSF + X2ndFlrSF
TotFlrSF <- trainExp$X1stFlrSF + trainExp$X2ndFlrSF
trainExp <- data.frame(trainExp, TotFlrSF)
# Remove outliers for TotFlrSF
# plot -> plot(y=trainExp$SalePrice, x=trainExp$TotFlrSF)
trainExp <- droplevels(subset(trainExp, trainExp$TotFlrSF != max(trainExp$TotFlrSF)))
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
# Remove points where Utilities is "NoSeWa"
trainExp <- droplevels(subset(trainExp, trainExp$Utilities == "AllPub"))
# Remove Utilities
trainExp <- trainExp[names(trainExp) != "Utilities"]

### set as factors
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
# NEW PREDICTOR: TotFlrSF <- X1stFlrSF + X2ndFlrSF
TotFlrSF <- testExp$X1stFlrSF + testExp$X2ndFlrSF
testExp <- data.frame(testExp, TotFlrSF)
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
### set as factors
testExp$remodeled <- as.factor(testExp$remodeled)
testExp$Bangalow <- as.factor(testExp$Bangalow)
testExp$Wooddeck <- as.factor(testExp$Wooddeck)
testExp$Pool <- as.factor(testExp$Pool)
testExp$Bsmt <- as.factor(testExp$Bsmt)

# Split up training set into training and testing sets
# First shuffle the data
# Split is around 75% training, 25% testing
train9 <- trainExp[sample(nrow(trainExp)),]
inTrain <- createDataPartition(y = train9$SalePrice, p = 0.75, list = FALSE)
trainTrain <- train9[inTrain,]
trainTest <- train9[-inTrain,]

all_levels <- TRUE
k <- 0
for (i in 1:length(trainTrain)) {
  if(is.factor(trainTrain[,i])) {
    if (!all(unique(train9[,i]) %in% unique(trainTrain[,i]))) {
      all_levels <- FALSE
      k <- i
      break
    }
  }
}
iterations <- 1
while (!all_levels) {
  iterations <- iterations + 1
  set.seed(iterations)
  train9 <- trainExp[sample(nrow(trainExp)),]
  inTrain <- createDataPartition(y = train9$SalePrice, p = 0.75, list = FALSE)
  trainTrain <- train9[inTrain,]
  trainTest <- train9[-inTrain,]
  
  all_levels <- TRUE
  for (i in 1:length(trainTrain)) {
    if(is.factor(trainTrain[,i])) {
      if (!all(unique(train9[,i]) %in% unique(trainTrain[,i]))) {
        all_levels <- FALSE
        break
      }
    }
  }
}

dim(trainTrain)
dim(trainTest)

# Only use the features with high visible correlation
feature9list <- c("LotArea", "OverallQual", "CentralAir", "TotFlrSF", "BedroomAbvGr", "WoodDeckSF")

Model9 <- lm(SalePrice ~ ., data = trainTrain[(names(trainTrain) != "Id") & ((names(trainTrain) == "SalePrice") | (names(trainTrain) %in% feature9list))])

Model9_trainTrain_SalePrice <- predict(Model9, newdata = data.frame(trainTrain[names(trainTrain) != "SalePrice"]))
Model9_trainTest_SalePrice <- predict(Model9, newdata = data.frame(trainTest[names(trainTest) != "SalePrice"]))

# Check RMSE
rmse(actual = trainTrain$SalePrice, predicted = Model9_trainTrain_SalePrice)
rmse(actual = trainTest$SalePrice, predicted = Model9_trainTest_SalePrice)


#####################################################################
### PREDICTIONS 10 Normalize/Scale features & transformation
#####################################################################
# Use the same feature extration as prediction 8
set.seed(1)

### Perform exploratory analysis with training data
trainExp <- train
# Remove training points with LotArea > 56600 (60000)
trainExp <- droplevels(subset(trainExp, trainExp$LotArea <= 60000))
# Remove some more training points for Lot Area
#trainExp <- droplevels(subset(trainExp, trainExp$LotArea <= 20000))
# Remove a few outliers with yearsbuilt 
trainExp <- droplevels(subset(trainExp, (trainExp$YearBuilt > 1900) | (trainExp$SalePrice < 200000)))
# NEW PREDICTOR: Remodeled -> whether property has been remodeled
remodeled <- trainExp$YearRemodAdd != trainExp$YearBuilt
trainExp <- data.frame(trainExp, remodeled)
# Remove heating = OthW, Floor from training set
trainExp <- droplevels(subset(trainExp, (trainExp$Heating != "OthW") & (trainExp$Heating != "Floor")))
# NEW PREDICTOR: TotFlrSF <- X1stFlrSF + X2ndFlrSF
TotFlrSF <- trainExp$X1stFlrSF + trainExp$X2ndFlrSF
trainExp <- data.frame(trainExp, TotFlrSF)
# Remove outliers for TotFlrSF
# plot -> plot(y=trainExp$SalePrice, x=trainExp$TotFlrSF)
trainExp <- droplevels(subset(trainExp, trainExp$TotFlrSF != max(trainExp$TotFlrSF)))
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
# Remove points where Utilities is "NoSeWa"
trainExp <- droplevels(subset(trainExp, trainExp$Utilities == "AllPub"))
# Remove Utilities
trainExp <- trainExp[names(trainExp) != "Utilities"]

### set as factors
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
# NEW PREDICTOR: TotFlrSF <- X1stFlrSF + X2ndFlrSF
TotFlrSF <- testExp$X1stFlrSF + testExp$X2ndFlrSF
testExp <- data.frame(testExp, TotFlrSF)
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
### set as factors
testExp$remodeled <- as.factor(testExp$remodeled)
testExp$Bangalow <- as.factor(testExp$Bangalow)
testExp$Wooddeck <- as.factor(testExp$Wooddeck)
testExp$Pool <- as.factor(testExp$Pool)
testExp$Bsmt <- as.factor(testExp$Bsmt)

# Find numeric features
colNum <- sapply(trainExp, is.numeric)
names(trainExp[,colNum])
# "Id"            "LotArea"       "YearBuilt"     "YearRemodAdd"  "X1stFlrSF"    
# "X2ndFlrSF"     "LowQualFinSF"  "GrLivArea"     "FullBath"      "HalfBath"     
# "BedroomAbvGr"  "KitchenAbvGr"  "TotRmsAbvGrd"  "Fireplaces"    "WoodDeckSF"   
# "OpenPorchSF"   "EnclosedPorch" "X3SsnPorch"    "ScreenPorch"   "PoolArea"     
# "MiscVal"       "YrSold"        "BsmtFinSF1"    "BsmtFinSF2"    "BsmtUnfSF"    
# "TotalBsmtSF"   "GarageCars"    "GarageArea"    "BsmtFullBath"  "BsmtHalfBath" 
# "MasVnrArea"    "GarageYrBlt"   "LotFrontage"   "SalePrice"     "TotFlrSF"  

# Find correlation
trainExpCorr <- trainExp[,colNum]
trainExpCorr <- trainExpCorr[names(trainExpCorr) != "Id"]
trainExpCorr <- trainExpCorr[names(trainExpCorr) != "SalePrice"]
corrNum <- cor(trainExpCorr, trainExpCorr)

corrThresh <- 0.2

pred1 <- as.character()
pred2 <- as.character()
corrCoef <- as.numeric()

for (i in 1:nrow(corrNum)) {
  len <- length(corrNum[i:nrow(corrNum),i][(corrNum[i:nrow(corrNum),i]>corrThresh | corrNum[i:nrow(corrNum),i]< -corrThresh) & (corrNum[i:nrow(corrNum),i] < max(corrNum[i:nrow(corrNum),i]))])
  if(len > 0) {
    for(k in 1:len) {
      pred1 <- c(pred1, names(corrNum[i:nrow(corrNum),i][corrNum[i:nrow(corrNum),i] == max(corrNum[i:nrow(corrNum),i])]))
      pred2 <- c(pred2, names(corrNum[i:nrow(corrNum),i][(corrNum[i:nrow(corrNum),i]>corrThresh | corrNum[i:nrow(corrNum),i]< -corrThresh) & (corrNum[i:nrow(corrNum),i] < max(corrNum[i:nrow(corrNum),i]))])[k])
      corrCoef <- c(corrCoef, corrNum[i:nrow(corrNum),i][(corrNum[i:nrow(corrNum),i]>corrThresh | corrNum[i:nrow(corrNum),i]< -corrThresh) & (corrNum[i:nrow(corrNum),i] < max(corrNum[i:nrow(corrNum),i]))][k])
    }
  }
}

corrNum.df <- data.frame(pred1 = pred1, pred2 = pred2, corrCoef = corrCoef)
corrNum.df <- corrNum.df[order(abs(corrNum.df$corrCoef),decreasing = TRUE),]


# Transform numeric features
plot(x=trainExp$LotArea, y=trainExp$SalePrice)
plot(x=trainExp$YearBuilt, y=trainExp$SalePrice)
plot(x=trainExp$YearRemodAdd, y=trainExp$SalePrice)
plot(x=trainExp$X1stFlrSF, y=trainExp$SalePrice)

# Normalizing numeric features



# Split up training set into training and testing sets
# First shuffle the data
# Split is around 75% training, 25% testing
train10 <- trainExp[sample(nrow(trainExp)),]
inTrain <- createDataPartition(y = train10$SalePrice, p = 0.75, list = FALSE)
trainTrain <- train10[inTrain,]
trainTest <- train10[-inTrain,]

all_levels <- TRUE
k <- 0
for (i in 1:length(trainTrain)) {
  if(is.factor(trainTrain[,i])) {
    if (!all(unique(train10[,i]) %in% unique(trainTrain[,i]))) {
      all_levels <- FALSE
      k <- i
      break
    }
  }
}
iterations <- 1
while (!all_levels) {
  iterations <- iterations + 1
  set.seed(iterations)
  train10 <- trainExp[sample(nrow(trainExp)),]
  inTrain <- createDataPartition(y = train10$SalePrice, p = 0.75, list = FALSE)
  trainTrain <- train10[inTrain,]
  trainTest <- train10[-inTrain,]
  
  all_levels <- TRUE
  for (i in 1:length(trainTrain)) {
    if(is.factor(trainTrain[,i])) {
      if (!all(unique(train10[,i]) %in% unique(trainTrain[,i]))) {
        all_levels <- FALSE
        break
      }
    }
  }
}

dim(trainTrain)
dim(trainTest)





