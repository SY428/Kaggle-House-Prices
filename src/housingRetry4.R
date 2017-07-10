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
library(plyr)
library(xgboost)

# Get files
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Set as appropriate (factor, num, etc)
train$SalePrice <- as.numeric(train$SalePrice)

# Add test "SalePrice"
test <- data.frame(test, SalePrice = as.numeric(rep(0,nrow(test))))

# Generate indexing to assign back to train and test later
train_id <- rep(TRUE, nrow(train))
test_id <- rep(FALSE, nrow(test))

dataFullA <- rbind(train,test)
dataFull <- data.frame(dataFullA[(names(dataFullA) != "SalePrice")], is_train = c(train_id,test_id))

# Check all non values
apply(dataFull,2,function(x) sum(x == ""))

# Set empty values to NAs
dataFull[dataFull == ""] <- NA

# Check all NAs in 
apply(dataFull,2,function(x) sum(is.na(x)))

########################################################################################################
### Removing Outliers
########################################################################################################
trainExp <- train
trainExp <- subset(trainExp, trainExp$LotArea <= 30000)

trainExp <- subset(trainExp, (trainExp$LotFrontage <= 150) | is.na(trainExp$LotFrontage))
trainExp <- subset(trainExp, trainExp$SalePrice <= 7e5)
plot(y=trainExp$SalePrice, x=trainExp$LotFrontage)
abline(lm(trainExp$SalePrice~trainExp$LotFrontage))

trainExp <- subset(trainExp, (trainExp$OverallCond != 2) | (trainExp$SalePrice < 3e5))
plot(trainExp$SalePrice~trainExp$OverallCond)
abline(lm(trainExp$SalePrice~trainExp$OverallCond))
# Bad line, but split later

trainExp <- subset(trainExp, (trainExp$GarageArea < 1200) | (trainExp$SalePrice > 3e5))

########################################################################################################
### Features Manipulation
### 1. New Features
### 1b. Nullify old Features
### 2. Set as factor/numeric features
### 3. Transform skewed features
### 4. Clean up some NAs
########################################################################################################
trainExp$LogSalePrice <- log(trainExp$SalePrice)
trainExp$MSSubClass <- as.factor(trainExp$MSSubClass) # Ordered factor

# trainExp$Residential <- ifelse(trainExp$MSZoning != "C (all)", 1, 0)
trainExp$MSZoning <- as.factor(trainExp$MSZoning) #OF

trainExp$LotFrontage <- as.numeric(trainExp$LotFrontage)
trainExp$LotArea <- as.numeric(trainExp$LotArea)
dataNonNAFrontage <- subset(trainExp, !is.na(trainExp$LotFrontage) & 
                              ((trainExp$LotArea < 17000) | (trainExp$LotFrontage > 90)) &
                              ((trainExp$LotArea < 16000) | (trainExp$LotFrontage > 85)) &
                              ((trainExp$LotArea < 8000) | (trainExp$LotFrontage > 40)) &
                              ((trainExp$LotArea < 12000) | (trainExp$LotFrontage > 57)) &
                              ((trainExp$LotArea < 14000) | (trainExp$LotFrontage > 70)) &
                              ((trainExp$LotArea < 20000) |  (trainExp$LotFrontage > 130)))
plot(y = dataNonNAFrontage$LotFrontage, x = dataNonNAFrontage$LotArea)
LotModel <- lm(dataNonNAFrontage$LotFrontage ~ dataNonNAFrontage$LotArea)
abline(LotModel)
predFrontage <- LotModel$coefficients[1] + trainExp$LotArea[which(is.na(trainExp$LotFrontage))]*LotModel$coefficients[2]
trainExp$LotFrontage[which(is.na(trainExp$LotFrontage))] <- predFrontage
plot(y = trainExp$LotFrontage, x = trainExp$LotArea)
####
par(mfrow=c(2,2))
regmodel <- lm(trainExp$SalePrice~trainExp$LotArea)
sqrmodel <- lm(trainExp$SalePrice~sqrt(trainExp$LotArea))
sqrsqrmodel <- lm(trainExp$SalePrice~sqrt(sqrt(trainExp$LotArea)))
plot(regmodel)
plot(sqrmodel)
plot(sqrsqrmodel)
logregmodel <- lm(log(trainExp$SalePrice)~trainExp$LotArea)
logsqrmodel <- lm(log(trainExp$SalePrice)~sqrt(trainExp$LotArea))
logsqrsqrmodel <- lm(log(trainExp$SalePrice)~sqrt(sqrt(trainExp$LotArea)))
plot(logregmodel)
plot(logsqrmodel)
plot(logsqrsqrmodel)
par(mfrow=c(1,1))
trainExp$LotAreaSqr <- sqrt(trainExp$LotArea)
trainExp$LotArea <- NULL

trainExp$StreetPave <- ifelse(trainExp$Street == "Pave", 1, 0)
trainExp$StreetPave <- as.factor(trainExp$StreetPave)
trainExp$Street <- NULL

#trainExp$AlleyNA <- ifelse(is.na(trainExp$Alley), 1, 0)
#trainExp$AlleyNA <- as.factor(trainExp$AlleyNA)
#trainExp$AlleyPave <- ifelse((trainExp$Alley == "Pave") & !is.na(trainExp$Alley), 1, 0)
#trainExp$AlleyPave <- as.factor(trainExp$AlleyPave)
#trainExp$Alley <- NULL
levels(trainExp$Alley) <- c(levels(trainExp$Alley), "NA")
trainExp$Alley[which(is.na(trainExp$Alley))] <- "NA"
trainExp$Alley <- as.factor(trainExp$Alley) # OF

#trainExp$LotShapeReg <- ifelse(trainExp$LotShape == "Reg", 1, 0)
#trainExp$LotShape <- NULL
trainExp$LotShape <- as.factor(trainExp$LotShape) # OF

#trainExp$LandContourHLS <- ifelse(trainExp$LandContour == "HLS", 1, 0)
#trainExp$LandContourLow <- ifelse(trainExp$LandContour == "Low", 1, 0)
#trainExp$LandContourLvl <- ifelse(trainExp$LandContour == "Lvl", 1, 0)
#trainExp$LandContour <- NULL
trainExp$LandContour <- as.factor(trainExp$LandContour) # OF

trainExp$Utilities <- NULL

#trainExp$LandSlopeMod <- ifelse(trainExp$LandSlope == "Mod", 1, 0)
#trainExp$LandSlopeSev <- ifelse(trainExp$LandSlope == "Sev", 1, 0)
#trainExp$LandSlope <- NULL
trainExp$LandSlope <- as.factor(trainExp$LandSlope) #OF

trainExp$LotConfig <- as.factor(trainExp$LotConfig) # Ordered Factor

trainExp$Neighborhood <- as.factor(trainExp$Neighborhood) # Ordered Factor

trainExp$Condition1 <- as.factor(trainExp$Condition1) # Ordered Factor

trainExp$Condition2 <- as.factor(trainExp$Condition2) # OF

#trainExp$BldgType1Fam <- ifelse(trainExp$BldgType == "1Fam", 1, 0)
#trainExp$BldgType <- NULL
trainExp$BldgType <- as.factor(trainExp$BldgType) # OF

#trainExp$HouseStyleHiPrice <- ifelse((trainExp$HouseStyle == "2.5Fin") | (trainExp$HouseStyle == "2Story"), 1, 0)
#trainExp$HouseStyleAbAvPrice <- ifelse((trainExp$HouseStyle == "1Story") | (trainExp$HouseStyle == "SLvl"), 1, 0)
#trainExp$HouseStyleAvPrice <- ifelse((trainExp$HouseStyle == "1.5Fin") | (trainExp$HouseStyle == "2.5Unf") | (trainExp$HouseStyle == "SFoyer"), 1, 0)
#trainExp$HouseStyle <- NULL
trainExp$HouseStyle <- as.factor(trainExp$HouseStyle) #OF

trainExp$OverallQualSqr <- trainExp$OverallQual^2
trainExp$OverallQualSqr <- as.numeric(trainExp$OverallQualSqr)
# par(mfrow=c(2,1))
# plot(trainExp$SalePrice~trainExp$OverallQualSqr)
# abline(lm(trainExp$SalePrice~trainExp$OverallQualSqr))
# plot(trainExp$SalePrice~(trainExp$OverallQual))
# abline(lm(trainExp$SalePrice~trainExp$OverallQual))
# par(mfrow=c(1,1))
trainExp$OverallQual <- NULL

trainExp$OverallCondHigh <- ifelse(trainExp$OverallCond > 6.5, 1, 0)
# trainExp$OverallCondHigh <- as.numeric(trainExp$OverallCondHigh)
trainExp$OverallCondLow <- ifelse(trainExp$OverallCond < 4.5, 1, 0)
# trainExp$OverallCondLow <- as.numeric(trainExp$OverallCondLow)
trainExp$OverallCond <- NULL

trainExp$Remodeled <- ifelse(trainExp$YearBuilt == trainExp$YearRemodAdd, 1, 0)
trainExp$Remodeled <- as.factor(trainExp$Remodeled)
trainExp$YrSoldAfterRemodel <- trainExp$YrSold - trainExp$YearRemodAdd
trainExp$YrSoldAfterRemodel <- as.numeric(trainExp$YrSoldAfterRemodel)
trainExp$YearRemodAdd <- NULL

trainExp$YearBuilt <- as.numeric(trainExp$YearBuilt)

# trainExp$RoofStyle <- NULL
trainExp$RoofStyle <- as.factor(trainExp$RoofStyle) # OF

# trainExp$RoofMatl <- NULL
trainExp$RoofMatl <- as.factor(trainExp$RoofMatl) # OF

trainExp$Exterior1st <- as.factor(trainExp$Exterior1st) # Ordered Factors

trainExp$Exterior2nd <- as.factor(trainExp$Exterior2nd) # Ordered Factors
# Deal with the 2 NAs separately

trainExp$MasVnrArea[is.na(trainExp$MasVnrArea)] <- 0
trainExp$MasVnrAreaSqr <- sqrt(trainExp$MasVnrArea)
trainExp$MasVnrArea <- NULL
trainExp$MasVnrType[is.na(trainExp$MasVnrType)] <- "None" # Ordered Factors

trainExp$ExterQual <- as.factor(trainExp$ExterQual) # Ordered Factors

trainExp$ExterCond <- as.factor(trainExp$ExterCond) # Ordered Factors

trainExp$Foundation <- as.factor(trainExp$Foundation) # Ordered Factors

# Somehow combine the BSMT variables (NAs in test bathroom)
trainExp$BsmtHeight <- trainExp$BsmtQual
trainExp$BsmtQual <- NULL
#from <- c("NA", "Fa", "TA", "Gd", "Ex")
#to <- c(11.53, 11.63, 11.81, 12.16, 12.67) - 11.53
levels(trainExp$BsmtHeight) <- c(levels(trainExp$BsmtHeight), "NA")
trainExp$BsmtHeight[which(is.na(trainExp$BsmtHeight))] <- "NA"  
trainExp$BsmtHeight <- as.factor(trainExp$BsmtHeight) # Ordered Factors
#trainExp$BsmtHeight <- as.numeric(as.character(mapvalues(trainExp$BsmtHeight, to=to, from=from)))

trainExp$TotalBsmtBath <- trainExp$BsmtFullBath + (0.5 * trainExp$BsmtHalfBath)
#trainExp$BsmtFullBath <- NULL
#trainExp$BsmtHalfBath <- NULL

# trainExp$BsmtCond <- NULL
levels(trainExp$BsmtCond) <- c(levels(trainExp$BsmtCond), "NA")
trainExp$BsmtCond[which(is.na(trainExp$BsmtCond))] <- "NA"
trainExp$BsmtCond <- as.factor(trainExp$BsmtCond) # OF

# trainExp$BsmtExposure <- NULL
levels(trainExp$BsmtExposure) <- c(levels(trainExp$BsmtExposure), "NA")
trainExp$BsmtExposure[which(is.na(trainExp$BsmtExposure))] <- "NA"
trainExp$BsmtExposure <- as.factor(trainExp$BsmtExposure) # OF

# trainExp$BsmtFinType2 <- NULL
levels(trainExp$BsmtFinType2) <- c(levels(trainExp$BsmtFinType2), "NA")
trainExp$BsmtFinType2[which(is.na(trainExp$BsmtFinType2))] <- "NA"
trainExp$BsmtFinType2 <- as.factor(trainExp$BsmtFinType2) # OF

trainExp$BsmtFinType1 <- as.factor(trainExp$BsmtFinType1) # Ordered factor

trainExp$BsmtTotalFinSF <- trainExp$BsmtFinSF1 + trainExp$BsmtFinSF2
trainExp$BsmtTotalFinSFSq <- trainExp$BsmtTotalFinSF^2
trainExp$BsmtTotalFinSF <- NULL
trainExp$BsmtFinSF1 <- NULL
trainExp$BsmtFinSF2 <- NULL

# has NA
# trainExp$BsmtUnfSF

# trainExp$Heating <- NULL
trainExp$Heating <- as.factor(trainExp$Heating) # OF

trainExp$HeatingQC <- as.factor(trainExp$HeatingQC) # Ordered Factors

trainExp$CentralAirYes <- ifelse(trainExp$CentralAir == "Y", 1, 0)
trainExp$CentralAirYes <- as.factor(trainExp$CentralAirYes)
trainExp$CentralAir <- NULL

# trainExp$ElectricalSBrkr <- ifelse((trainExp$Electrical == "SBrkr") | is.na(trainExp$Electrical), 1, 0)
# trainExp$Electrical <- NULL
trainExp$Electrical[which(is.na(trainExp$Electrical))] <- "SBrkr"
trainExp$Electrical <- as.factor(trainExp$Electrical) # OF

plot(trainExp$SalePrice~trainExp$X1stFlrSF)
regModel <- lm(trainExp$SalePrice~trainExp$X1stFlrSF)
reglogModel <- lm(trainExp$SalePrice~log(trainExp$X1stFlrSF+1))
loglogModel <- lm(log(trainExp$SalePrice)~log(trainExp$X1stFlrSF+1))
logregModel <- lm(log(trainExp$SalePrice)~(trainExp$X1stFlrSF))
plot(trainExp$SalePrice~trainExp$X1stFlrSF)
abline(regModel)
plot(trainExp$SalePrice~log(trainExp$X1stFlrSF+1))
abline(reglogModel)
plot(log(trainExp$SalePrice)~log(trainExp$X1stFlrSF+1))
abline(loglogModel)
plot(log(trainExp$SalePrice)~(trainExp$X1stFlrSF))
abline(logregModel)
# par(mfrow=c(2,2))
# plot(regModel)
# plot(reglogModel)
# plot(loglogModel)
# plot(logregModel)
# par(mfrow=c(1,1))
trainExp$Log1stFlrSF <- log(trainExp$X1stFlrSF+1)
trainExp$X1stFlrSF <- NULL

trainExp$Has2Flrs <- ifelse(trainExp$X2ndFlrSF > 0, 1, 0)
trainExp$Has2Flrs <- as.factor(trainExp$Has2Flrs)
trainExp$X2ndFlrSF <- NULL

trainExp$LowQualFinSF <- NULL

trainExp$LogGrLivArea <- log(1+trainExp$GrLivArea)
trainExp$GrLivArea <- NULL

# Multiply Bsmt by 0.75
trainExp$TotalBath <- trainExp$FullBath + (0.5 * trainExp$HalfBath) +
  0.75 * (trainExp$BsmtFullBath + (0.5 * trainExp$BsmtHalfBath))
plot(log(trainExp$SalePrice)~log(1+trainExp$TotalBath))
logModel <- (lm(log(trainExp$SalePrice)~log(1+trainExp$TotalBath)))
abline(logModel)
plot(log(trainExp$SalePrice)~(trainExp$TotalBath))
regModel <- (lm(log(trainExp$SalePrice)~(trainExp$TotalBath)))
abline(regModel)

trainExp$LogTotalBath <- log(1+trainExp$TotalBath)
trainExp$TotalBath <- NULL
trainExp$FullBath <- NULL
trainExp$HalfBath <- NULL
trainExp$BsmtFullBath <- NULL
trainExp$BsmtHalfBath <- NULL

trainExp$BedroomAbvGr <- NULL

trainExp$KitchenAbvGr <- NULL

trainExp$KitchenQual <- as.factor(trainExp$KitchenQual) # Ordered Factor

plot(log(trainExp$SalePrice)~trainExp$TotRmsAbvGrd)
logregModel <- lm(log(trainExp$SalePrice)~trainExp$TotRmsAbvGrd)
abline(logregModel)
plot(log(trainExp$SalePrice)~log(trainExp$TotRmsAbvGrd))
loglogModel <- lm(log(trainExp$SalePrice)~log(trainExp$TotRmsAbvGrd))
abline(loglogModel)
trainExp$LogTotRmsAbvGrd <- log(trainExp$TotRmsAbvGrd)
trainExp$TotRmsAbvGrd <- NULL

trainExp$Functional <- as.factor(trainExp$Functional) # Ordered Factor

plot(log(trainExp$SalePrice)~((trainExp$Fireplaces)))
logregModel <- lm(log(trainExp$SalePrice)~((trainExp$Fireplaces)))
abline(logregModel)
plot(log(trainExp$SalePrice)~(sqrt(trainExp$Fireplaces)))
logsqrModel <- lm(log(trainExp$SalePrice)~(sqrt(trainExp$Fireplaces)))
abline(logsqrModel)
plot(log(trainExp$SalePrice)~sqrt(sqrt(trainExp$Fireplaces)))
logsqr2Model <- lm(log(trainExp$SalePrice)~sqrt(sqrt(trainExp$Fireplaces)))
abline(logsqr2Model)
trainExp$FireplacesSqr2 <- sqrt(sqrt(trainExp$Fireplaces))
trainExp$Fireplaces <- NULL

trainExp$FireplaceQu <- as.factor(trainExp$FireplaceQu) # Ordered Factor

# trainExp$GarageType <- NULL
levels(trainExp$GarageType) <- c(levels(trainExp$GarageType), "NA")
trainExp$GarageType[which(is.na(trainExp$GarageType))] <- "NA"
trainExp$GarageType <- as.factor(trainExp$GarageType) # OF

trainExp$HasGarage <- ifelse(!is.na(trainExp$GarageYrBlt), 1, 0)
trainExp$HasGarage <- as.factor(trainExp$HasGarage)
# trainExp$GarageRebuilt <- ifelse(trainExp$GarageYrBlt != trainExp$YearBuilt, 1, 0)
trainExp$GarageYrBlt <- NULL

# trainExp$GarageFinish <- NULL
levels(trainExp$GarageFinish) <- c(levels(trainExp$GarageFinish), "NA")
trainExp$GarageFinish[which(is.na(trainExp$GarageFinish))] <- "NA"
trainExp$GarageFinish <- as.factor(trainExp$GarageFinish)

trainExp$GarageCars <- as.numeric(trainExp$GarageCars)

plot(trainExp$LogSalePrice~(trainExp$GarageArea))
logregModel <- lm(trainExp$LogSalePrice~trainExp$GarageArea)
abline(logregModel)
trainExp$GarageArea <- as.numeric(trainExp$GarageArea)

# trainExp$GarageCond <- NULL
levels(trainExp$GarageCond) <- c(levels(trainExp$GarageCond), "NA")
trainExp$GarageCond[which(is.na(trainExp$GarageCond))] <- "NA"
trainExp$GarageCond <- as.factor(trainExp$GarageCond) # OF

trainExp$GarageQual <- as.factor(trainExp$GarageQual) # Ordered Factor

trainExp$PavedDrive <- as.factor(trainExp$PavedDrive) # Ordered Factor

trainExp$TotalPorchDeckArea <- trainExp$WoodDeckSF + trainExp$OpenPorchSF + trainExp$EnclosedPorch + trainExp$X3SsnPorch + trainExp$ScreenPorch
# plot(trainExp$LogSalePrice~(trainExp$TotalPorchDeckArea))
# logregModel <- lm(trainExp$LogSalePrice~(trainExp$TotalPorchDeckArea))
# abline(logregModel)
trainExp <- subset(trainExp, (trainExp$LogSalePrice > 11) | (trainExp$TotalPorchDeckArea < 200))
plot(trainExp$LogSalePrice~(trainExp$TotalPorchDeckArea))
logregModel <- lm(trainExp$LogSalePrice~(trainExp$TotalPorchDeckArea))
abline(logregModel)
trainExp$WoodDeckSF <- NULL
trainExp$OpenPorchSF <- NULL
trainExp$EnclosedPorch <- NULL
trainExp$X3SsnPorch <- NULL
trainExp$ScreenPorch <- NULL

trainExp$HasPool <- ifelse(trainExp$PoolArea > 0, 1, 0)
trainExp$HasPool <- as.factor(trainExp$HasPool)
trainExp$PoolArea <- NULL

trainExp$PoolQC <- NULL

# trainExp$HasFence <- ifelse(!is.na(trainExp$Fence), 1, 0)
# trainExp$Fence <- NULL
levels(trainExp$Fence) <- c(levels(trainExp$Fence), "NA")
trainExp$Fence[which(is.na(trainExp$Fence))] <- "NA"
trainExp$Fence <- as.factor(trainExp$Fence) # OF

trainExp$MiscFeature <- NULL

trainExp$MiscVal <- NULL

# trainExp$MoSold <- NULL
trainExp$MoSold <- as.factor(trainExp$MoSold) # OF

# trainExp$YrSold <- NULL
trainExp$YrSold <- as.factor(trainExp$YrSold) # OF

trainExp$NewSale <- ifelse(trainExp$SaleType == "New", 1, 0)
trainExp$NewSale <- as.factor(trainExp$NewSale)
# trainExp$ConSale <- ifelse((trainExp$SaleType == "Con") | (trainExp$SaleType == "ConLD") | (trainExp$SaleType == "ConLI"), 1, 0)
# trainExp$SaleType <- NULL
trainExp$SaleType <- as.factor(trainExp$SaleType) # OF

# trainExp$SaleCondPartial <- ifelse(trainExp$SaleCondition == "Partial", 1, 0)
# trainExp$SaleCondNormal <- ifelse(trainExp$SaleCondition == "Normal", 1, 0)
# trainExp$SaleCondition <- NULL
trainExp$SaleCondition <- as.factor(trainExp$SaleCondition) # OF

### Deal with ordered factors
# ordered.factors <- c("MSSubClass", "LotConfig", "Neighborhood", "Condition1", 
#                      "ExterQual", "ExterCond", "Foundation", "HeatingQC", "PavedDrive",
#                      "MasVnrType",
#                      
#                      "Exterior1st", "Exterior2nd", "BsmtFinType1", "KitchenQual",
#                      "Functional", "FireplaceQu", "GarageQual")

# Make ordered factors continuous
# First remove NAs

# test has Exterior1st NAs
# test has Exterior2nd NAs

levels(trainExp$BsmtFinType1) <- c(levels(trainExp$BsmtFinType1), "NA")
trainExp$BsmtFinType1[which(is.na(trainExp$BsmtFinType1))] <- "NA"

# test has KitchenQual NAs
# test has Functional NAs

levels(trainExp$FireplaceQu) <- c(levels(trainExp$FireplaceQu), "NA")
trainExp$FireplaceQu[which(is.na(trainExp$FireplaceQu))] <- "NA"

levels(trainExp$GarageQual) <- c(levels(trainExp$GarageQual), "NA")
trainExp$GarageQual[which(is.na(trainExp$GarageQual))] <- "NA"

ordered.factors <- c()

for (i in 1:ncol(trainExp)) {
  if(is.factor(trainExp[,i]) & (length(levels(trainExp[,i])) > 2)) {
    ordered.factors <- c(ordered.factors, colnames(trainExp)[i])
  }
}

################################
# Do the same with testing data
# Clean up data
################################
testExp <- test
testExp$LogSalePrice <- 0

sqrt(testExp[which(testExp$MSSubClass==150),]$LotArea)
plot(trainExp$LotAreaSqr~trainExp$MSSubClass)
plot(trainExp$YearBuilt~trainExp$MSSubClass)

# 180 is the closest 
testExp$MSSubClass[which(testExp$MSSubClass==150)] <- 180
testExp$MSSubClass <- as.factor(testExp$MSSubClass) # Ordered factor

# testExp$Residential <- ifelse(testExp$MSZoning != "C (all)" | is.na(testExp$MSZoning), 1, 0)
# testExp$Residential <- as.factor(testExp$Residential)
# testExp$MSZoning <- NULL
# Assign to approximate 456  757  791 1445
testExp$MSZoning[456] <- "RH"
testExp$MSZoning[757] <- "RH"
testExp$MSZoning[791] <- "RM"
testExp$MSZoning[1445] <- "RH"
testExp$MSZoning <- as.factor(testExp$MSZoning) # OF

testExp$LotFrontage <- as.numeric(testExp$LotFrontage)
testExp$LotArea <- as.numeric(testExp$LotArea)

predFrontage <- LotModel$coefficients[1] + testExp$LotArea[which(is.na(testExp$LotFrontage))]*LotModel$coefficients[2]

testExp$LotFrontage[which(is.na(testExp$LotFrontage))] <- predFrontage
plot(y = testExp$LotFrontage, x = testExp$LotArea)

testExp$LotAreaSqr <- sqrt(testExp$LotArea)
testExp$LotArea <- NULL

testExp$StreetPave <- ifelse(testExp$Street == "Pave", 1, 0)
testExp$StreetPave <- as.factor(testExp$StreetPave)
testExp$Street <- NULL

# testExp$AlleyNA <- ifelse(is.na(testExp$Alley), 1, 0)
# testExp$AlleyPave <- ifelse((testExp$Alley == "Pave") & !is.na(testExp$Alley), 1, 0)
# testExp$Alley <- NULL
levels(testExp$Alley) <- c(levels(testExp$Alley), "NA")
testExp$Alley[which(is.na(testExp$Alley))] <- "NA"
testExp$Alley <- as.factor(testExp$Alley) # OF

# testExp$LotShapeReg <- ifelse(testExp$LotShape == "Reg", 1, 0)
# testExp$LotShape <- NULL
testExp$LotShape <- as.factor(testExp$LotShape) # OF

# testExp$LandContourHLS <- ifelse(testExp$LandContour == "HLS", 1, 0)
# testExp$LandContourLow <- ifelse(testExp$LandContour == "Low", 1, 0)
# testExp$LandContourLvl <- ifelse(testExp$LandContour == "Lvl", 1, 0)
# testExp$LandContour <- NULL
testExp$LandContour <- as.factor(testExp$LandContour) # OF

testExp$Utilities <- NULL

# testExp$LandSlopeMod <- ifelse(testExp$LandSlope == "Mod", 1, 0)
# testExp$LandSlopeSev <- ifelse(testExp$LandSlope == "Sev", 1, 0)
# testExp$LandSlope <- NULL
testExp$LandSlope <- as.factor(testExp$LandSlope) # OF

testExp$LotConfig <- as.factor(testExp$LotConfig) # Ordered Factor

testExp$Neighborhood <- as.factor(testExp$Neighborhood) # Ordered Factor

testExp$Condition1 <- as.factor(testExp$Condition1) # Ordered Factor

testExp$Condition2 <- as.factor(testExp$Condition2) # OF

# testExp$BldgType1Fam <- ifelse(testExp$BldgType == "1Fam", 1, 0)
# testExp$BldgType <- NULL
testExp$BldgType <- as.factor(testExp$BldgType) # OF

# testExp$HouseStyleHiPrice <- ifelse((testExp$HouseStyle == "2.5Fin") | (testExp$HouseStyle == "2Story"), 1, 0)
# testExp$HouseStyleAbAvPrice <- ifelse((testExp$HouseStyle == "1Story") | (testExp$HouseStyle == "SLvl"), 1, 0)
# testExp$HouseStyleAvPrice <- ifelse((testExp$HouseStyle == "1.5Fin") | (testExp$HouseStyle == "2.5Unf") | (testExp$HouseStyle == "SFoyer"), 1, 0)
# testExp$HouseStyleLowPrice <- ifelse(testExp$HouseStyle == "1.5Unf", 1, 0)
# testExp$HouseStyle <- NULL
testExp$HouseStyle <- as.factor(testExp$HouseStyle) # OF

testExp$OverallQualSqr <- testExp$OverallQual^2
testExp$OverallQualSqr <- as.numeric(testExp$OverallQualSqr)
testExp$OverallQual <- NULL

testExp$OverallCondHigh <- ifelse(testExp$OverallCond > 6.5, 1, 0)
# testExp$OverallCondHigh <- as.numeric(testExp$OverallCondHigh)
testExp$OverallCondLow <- ifelse(testExp$OverallCond < 4.5, 1, 0)
# testExp$OverallCondLow <- as.numeric(testExp$OverallCondLow)
testExp$OverallCond <- NULL

testExp$Remodeled <- ifelse(testExp$YearBuilt == testExp$YearRemodAdd, 1, 0)
testExp$Remodeled <- as.factor(testExp$Remodeled)
testExp$YrSoldAfterRemodel <- testExp$YrSold - testExp$YearRemodAdd
testExp$YrSoldAfterRemodel <- as.numeric(testExp$YrSoldAfterRemodel)
testExp$YearRemodAdd <- NULL

testExp$YearBuilt <- as.numeric(testExp$YearBuilt)

# testExp$RoofStyle <- NULL
testExp$RoofStyle <- as.factor(testExp$RoofStyle) # OF

# testExp$RoofMatl <- NULL
testExp$RoofMatl <- as.factor(testExp$RoofMatl) # OF

testExp$Exterior1st <- as.factor(testExp$Exterior1st) # Ordered Factors

testExp$Exterior2nd <- as.factor(testExp$Exterior2nd) # Ordered Factors
# Deal with the 2 NAs separately

testExp$MasVnrArea[is.na(testExp$MasVnrArea)] <- 0
testExp$MasVnrAreaSqr <- sqrt(testExp$MasVnrArea)
testExp$MasVnrArea <- NULL
testExp$MasVnrType[is.na(testExp$MasVnrType)] <- "None"

testExp$ExterQual <- as.factor(testExp$ExterQual) # Ordered Factors

testExp$ExterCond <- as.factor(testExp$ExterCond) # Ordered Factors

testExp$Foundation <- as.factor(testExp$Foundation) # Ordered Factors

# Somehow combine the BSMT variables (NAs in test bathroom)
testExp$BsmtHeight <- testExp$BsmtQual
testExp$BsmtQual <- NULL
# from <- c("NA", "Fa", "TA", "Gd", "Ex")
# to <- c(11.53, 11.63, 11.81, 12.16, 12.67) - 11.53
levels(testExp$BsmtHeight) <- c("Ex", "Fa", "Gd", "TA", "NA")
testExp$BsmtHeight[which(is.na(testExp$BsmtHeight))] <- "NA"
testExp$BsmtHeight <- as.factor(testExp$BsmtHeight) # Ordered Factors
# testExp$BsmtHeight <- as.numeric(as.character(mapvalues(testExp$BsmtHeight, to=to, from=from)))

# NAs in bathrooms set to 0
testExp$BsmtFullBath[which(is.na(testExp$BsmtFullBath))] <- 0
testExp$BsmtHalfBath[which(is.na(testExp$BsmtHalfBath))] <- 0
testExp$TotalBsmtBath <- testExp$BsmtFullBath + (0.5 * testExp$BsmtHalfBath)

# testExp$BsmtCond <- NULL
levels(testExp$BsmtCond) <- c(levels(testExp$BsmtCond), "NA")
testExp$BsmtCond[which(is.na(testExp$BsmtCond))] <- "NA"
testExp$BsmtCond <- as.factor(testExp$BsmtCond) # OF

# testExp$BsmtExposure <- NULL
levels(testExp$BsmtExposure) <- c(levels(testExp$BsmtExposure), "NA")
testExp$BsmtExposure[which(is.na(testExp$BsmtExposure))] <- "NA"
testExp$BsmtExposure <- as.factor(testExp$BsmtExposure) # OF

# testExp$BsmtFinType2 <- NULL
levels(testExp$BsmtFinType2) <- c(levels(testExp$BsmtFinType2), "NA")
testExp$BsmtFinType2[which(is.na(testExp$BsmtFinType2))] <- "NA"
testExp$BsmtFinType2 <- as.factor(testExp$BsmtFinType2) # OF

testExp$BsmtFinType1 <- as.factor(testExp$BsmtFinType1) # Ordered factor

testExp$BsmtFinSF1[which(is.na(testExp$BsmtFinSF1))] <- 0
testExp$BsmtFinSF2[which(is.na(testExp$BsmtFinSF2))] <- 0
testExp$BsmtTotalFinSF <- testExp$BsmtFinSF1 + testExp$BsmtFinSF2
testExp$BsmtTotalFinSFSq <- testExp$BsmtTotalFinSF^2
testExp$BsmtTotalFinSF <- NULL
testExp$BsmtFinSF1 <- NULL
testExp$BsmtFinSF2 <- NULL

# has NA
testExp$TotalBsmtSF[which(is.na(testExp$TotalBsmtSF))] <- 0

# has NA
testExp$BsmtUnfSF[which(is.na(testExp$BsmtUnfSF))] <- 0

# testExp$Heating <- NULL
testExp$Heating <- as.factor(testExp$Heating) # OF

testExp$HeatingQC <- as.factor(testExp$HeatingQC) # Ordered Factors

testExp$CentralAirYes <- ifelse(testExp$CentralAir == "Y", 1, 0)
testExp$CentralAirYes <- as.factor(testExp$CentralAirYes)
testExp$CentralAir <- NULL

# testExp$ElectricalSBrkr <- ifelse((testExp$Electrical == "SBrkr") | is.na(testExp$Electrical), 1, 0)
# testExp$Electrical <- NULL
testExp$Electrical <- as.factor(testExp$Electrical) # OF

testExp$Log1stFlrSF <- log(testExp$X1stFlrSF+1)
testExp$X1stFlrSF <- NULL

testExp$Has2Flrs <- ifelse(testExp$X2ndFlrSF > 0, 1, 0)
testExp$Has2Flrs <- as.factor(testExp$Has2Flrs)
testExp$X2ndFlrSF <- NULL

testExp$LowQualFinSF <- NULL

testExp$LogGrLivArea <- log(1+testExp$GrLivArea)
testExp$GrLivArea <- NULL

# Multiply Bsmt by 0.75
testExp$TotalBath <- testExp$FullBath + (0.5 * testExp$HalfBath) +
  0.75 * (testExp$BsmtFullBath + (0.5 * testExp$BsmtHalfBath))

testExp$LogTotalBath <- log(1+testExp$TotalBath)
testExp$TotalBath <- NULL
testExp$FullBath <- NULL
testExp$HalfBath <- NULL
testExp$BsmtFullBath <- NULL
testExp$BsmtHalfBath <- NULL

testExp$BedroomAbvGr <- NULL

testExp$KitchenAbvGr <- NULL

testExp$KitchenQual <- as.factor(testExp$KitchenQual) # Ordered Factor

testExp$LogTotRmsAbvGrd <- log(testExp$TotRmsAbvGrd)
testExp$TotRmsAbvGrd <- NULL

testExp$Functional <- as.factor(testExp$Functional) # Ordered Factor

testExp$FireplacesSqr2 <- sqrt(sqrt(testExp$Fireplaces))
testExp$Fireplaces <- NULL

testExp$FireplaceQu <- as.factor(testExp$FireplaceQu) # Ordered Factor

# testExp$GarageType <- NULL
levels(testExp$GarageType) <- c(levels(testExp$GarageType), "NA")
testExp$GarageType[which(is.na(testExp$GarageType))] <- "NA"
testExp$GarageType <- as.factor(testExp$GarageType) # OF

which(testExp$GarageYrBlt>2010)
testExp$GarageYrBlt[1133] <- 2007
testExp$HasGarage <- ifelse(!is.na(testExp$GarageYrBlt), 1, 0)
testExp$HasGarage <- as.factor(testExp$HasGarage)
# testExp$GarageRebuilt <- ifelse(testExp$GarageYrBlt != testExp$YearBuilt, 1, 0)
testExp$GarageYrBlt <- NULL

# testExp$GarageFinish <- NULL
levels(testExp$GarageFinish) <- c(levels(testExp$GarageFinish), "NA")
testExp$GarageFinish[which(is.na(testExp$GarageFinish))] <- "NA"
testExp$GarageFinish <- as.factor(testExp$GarageFinish)

testExp$GarageCars[which(is.na(testExp$GarageCars))] <- 0
testExp$GarageCars <- as.numeric(testExp$GarageCars)

testExp$GarageArea[which(is.na(testExp$GarageArea))] <- 0
testExp$GarageArea <- as.numeric(testExp$GarageArea)

# testExp$GarageCond <- NULL
levels(testExp$GarageCond) <- c(levels(testExp$GarageCond), "NA")
testExp$GarageCond[which(is.na(testExp$GarageCond))] <- "NA"
testExp$GarageCond <- as.factor(testExp$GarageCond) # OF

testExp$GarageQual <- as.factor(testExp$GarageQual) # Ordered Factor

testExp$PavedDrive <- as.factor(testExp$PavedDrive) # Ordered Factor

testExp$TotalPorchDeckArea <- testExp$WoodDeckSF + testExp$OpenPorchSF + testExp$EnclosedPorch + testExp$X3SsnPorch + testExp$ScreenPorch
testExp$WoodDeckSF <- NULL
testExp$OpenPorchSF <- NULL
testExp$EnclosedPorch <- NULL
testExp$X3SsnPorch <- NULL
testExp$ScreenPorch <- NULL

testExp$HasPool <- ifelse(testExp$PoolArea > 0, 1, 0)
testExp$HasPool <- as.factor(testExp$HasPool)
testExp$PoolArea <- NULL

testExp$PoolQC <- NULL

# testExp$HasFence <- ifelse(!is.na(testExp$Fence), 1, 0)
# testExp$Fence <- NULL
levels(testExp$Fence) <- c(levels(testExp$Fence), "NA")
testExp$Fence[which(is.na(testExp$Fence))] <- "NA"
testExp$Fence <- as.factor(testExp$Fence) # OF

testExp$MiscFeature <- NULL

testExp$MiscVal <- NULL

# testExp$MoSold <- NULL
testExp$MoSold <- as.factor(testExp$MoSold) # OF

# testExp$YrSold <- NULL
testExp$YrSold <- as.factor(testExp$YrSold) # OF

testExp$NewSale <- ifelse(testExp$SaleType == "New", 1, 0)
# testExp$ConSale <- ifelse((testExp$SaleType == "Con") | (testExp$SaleType == "ConLD") | (testExp$SaleType == "ConLI"), 1, 0)
# testExp$SaleType <- NULL
testExp$NewSale[1030] <- 0
# testExp$ConSale[1030] <- 0
testExp$NewSale <- as.factor(testExp$NewSale)
testExp$SaleType[1030] <- "COD"
testExp$SaleType <- as.factor(testExp$SaleType) # OF

# testExp$SaleCondPartial <- ifelse(testExp$SaleCondition == "Partial", 1, 0)
# testExp$SaleCondNormal <- ifelse(testExp$SaleCondition == "Normal", 1, 0)
# testExp$SaleCondition <- NULL
testExp$SaleCondition <- as.factor(testExp$SaleCondition) # OF

# Make ordered factors continuous
# First remove NAs

# test has Exterior1st NAs (most popular for 1940)
testExp$Exterior1st[which(is.na(testExp$Exterior1st))] <- "Wd Sdng"

# test has Exterior2nd NAs (most popular for 1940)
testExp$Exterior2nd[which(is.na(testExp$Exterior2nd))] <- "Wd Sdng"

levels(testExp$BsmtFinType1) <- c(levels(testExp$BsmtFinType1), "NA")
testExp$BsmtFinType1[which(is.na(testExp$BsmtFinType1))] <- "NA"

# test has KitchenQual NAs (year built is 1917)
testExp$KitchenQual[which(is.na(testExp$KitchenQual))] <- "Fa"

# test has Functional NAs (based vaguely on intuition)
which(is.na(testExp$Functional))
testExp$Functional[757] <- "Mod"
testExp$Functional[1014] <- "Typ"

levels(testExp$FireplaceQu) <- c(levels(testExp$FireplaceQu), "NA")
testExp$FireplaceQu[which(is.na(testExp$FireplaceQu))] <- "NA"

levels(testExp$GarageQual) <- c(levels(testExp$GarageQual), "NA")
testExp$GarageQual[which(is.na(testExp$GarageQual))] <- "NA"

########################################################################################################
### Deal with Ordered Factors
########################################################################################################

trainExpRemove <- trainExp[names(trainExp) != "Id" & names(trainExp) != "SalePrice"]

LMtrainModel <- lm(LogSalePrice~., trainExpRemove)
LMtrainSelModel0 <- lm(LogSalePrice ~ YearBuilt + LotAreaSqr + OverallQualSqr 
                       + OverallCondHigh + OverallCondLow + YrSoldAfterRemodel
                       + LogGrLivArea + TotalPorchDeckArea, 
                       trainExp)
LMtrainSelModel1 <- lm(LogSalePrice ~ YearBuilt + LotAreaSqr + OverallQualSqr 
                       + OverallCondHigh + OverallCondLow + LogGrLivArea + TotalPorchDeckArea, 
                       trainExp)

## Deal with ordered factors

BaseModel <- "LogSalePrice ~ YearBuilt + LotAreaSqr + OverallQualSqr + OverallCondHigh + OverallCondLow + LogGrLivArea + TotalPorchDeckArea"

MakeFactorContinuous <- function(df, factors, bm) {
  factor_list <- list()
  basemodel <- bm
  for (i in 1:length(factors)) {
    formula <- paste(basemodel, " + ", factors[i], " -1", sep = "")
    new_model <- lm(formula = formula, data = df)
    coef_ind <- grep(factors[i], names(new_model$coefficients))
    # print(summary(new_model))
    
    from <- gsub(factors[i], "", names(new_model$coefficients[coef_ind]))
    to <- as.numeric(new_model$coefficients[coef_ind])
    # print(from)
    # print(to)
    
    df[[factors[i]]] <- as.numeric(as.character(mapvalues(df[[factors[i]]], to=to, from=from)))
    
    factor_list[length(factor_list)+1] <- list(list(from,to))
    basemodel <- paste(basemodel, " + ", factors[i], sep = "")
  }
  # To access list of list
  # factor_list[[factorNum]][[factorFrom or factorTo]]
  list(df, factor_list)
}


# Convert training data into continuous
df_factor_list <- MakeFactorContinuous(df = trainExp, factors = ordered.factors, bm = BaseModel)

trainExpCont <- df_factor_list[[1]]
factor_list <- df_factor_list[[2]]

LMtrainContModel <- lm(LogSalePrice~. - Id - SalePrice, trainExpCont)

#######################################################################
### Convert test ordered factors the same way
#######################################################################

MakeTestFactorContinuous <- function(df, factors, factor_list) {
  for (i in 1:length(factors)) {
    from <- factor_list[[i]][[1]]
    to <- factor_list[[i]][[2]]
    
    df[[factors[i]]] <- as.numeric(as.character(mapvalues(df[[factors[i]]], to=to, from=from)))
  }
  df
}

testExpCont <- MakeTestFactorContinuous(df = testExp, factors = ordered.factors, factor_list)
# Warning for "from" is okay

myRMSLE <- function(actual, predicted) {
  sqrt(1/(length(actual)) * sum((log(actual)-log(predicted))^2))
}

# Remove SalePrice and Id from trainExpCont and testExpCont
trainExpCont_SalePrice <- trainExpCont$SalePrice
trainExpCont_Id <- trainExpCont$Id

testExpCont_Id <- testExpCont$Id

trainExpCont$SalePrice <- NULL
trainExpCont$Id <- NULL
trainExpCont$OverallCondHigh <- as.factor(trainExpCont$OverallCondHigh)
trainExpCont$OverallCondLow <- as.factor(trainExpCont$OverallCondLow)
# trainExpCont$Residential <- as.numeric(trainExpCont$Residential)

testExpCont$SalePrice <- NULL
testExpCont$Id <- NULL
testExpCont$OverallCondHigh <- as.factor(testExpCont$OverallCondHigh)
testExpCont$OverallCondLow <- as.factor(testExpCont$OverallCondLow)
# testExpCont$Residential <- as.numeric(testExpCont$Residential)

LMtrainContModel2 <- lm(LogSalePrice~., trainExpCont)

############################
## Tweak Existing Predictors
############################

# feature_removed <- c()
# feature_removed <- c("LotFrontage", "FireplaceQu", "LandContourLvl", 
#                      "HouseStyleHiPrice", "HouseStyleAvPrice", "Has2Flrs", 
#                      "SaleCondPartial")
# trainExpCont <- trainExpCont[!(names(trainExpCont) %in% feature_removed)]
# testExpCont <- testExpCont[!(names(testExpCont) %in% feature_removed)]

# LMtrainContModel3 <- lm(LogSalePrice~., trainExpCont)

############################
## Find Correlation
############################

trainExpContNum <- trainExpCont[names(trainExpCont) != "LogSalePrice"]

for (i in 1: ncol(trainExpContNum)) {
  trainExpContNum[[names(trainExpContNum)[i]]] <- as.numeric(as.character(trainExpContNum[[names(trainExpContNum)[i]]]))
}

testExpContNum <- testExpCont[names(testExpCont) != "LogSalePrice"]

for (i in 1: ncol(testExpContNum)) {
  testExpContNum[[names(testExpContNum)[i]]] <- as.numeric(as.character(testExpContNum[[names(testExpContNum)[i]]]))
}

corrNum <- cor(trainExpContNum, trainExpContNum)

corrThresh <- 0.3

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

# modelFormula <- "LogSalePrice ~ "
# for (i in 1:ncol(trainExpCont[names(trainExpCont) != "LogSalePrice"])) {
#   if (i < ncol(trainExpCont[names(trainExpCont) != "LogSalePrice"])) {
#     modelFormula <- paste(modelFormula, names(trainExpCont[names(trainExpCont) != "LogSalePrice"])[i], " + ", sep = "")
#   } else {
#     modelFormula <- paste(modelFormula, names(trainExpCont[names(trainExpCont) != "LogSalePrice"])[i], sep = "")
#   }
# }
# 
# modelFormulaCorr <- modelFormula
# for (i in 1:nrow(corrNum.df)) {
#   modelFormulaCorr <- paste(modelFormulaCorr, " + ", corrNum.df$pred1[i], "*", corrNum.df$pred2[i], sep = "")
# }
# 

FormulaCorr <- ""
for (i in 1:nrow(corrNum.df)) {
  FormulaCorr <- paste(FormulaCorr, corrNum.df$pred1[i], "*", corrNum.df$pred2[i], " + ", sep = "")
}
FormulaCorr
#######################################################################
### Use PCA on existing features to remove any correlations
#######################################################################

# preProc <- preProcess(trainExpCont[names(trainExpCont) != "LogSalePrice"], method = "pca", thresh = 0.99)
# trainPCA <- predict(preProc, trainExpCont[names(trainExpCont) != "LogSalePrice"])
# trainPCA <- data.frame(trainPCA, LogSalePrice = trainExpCont$LogSalePrice)
# 
# testPCA <- predict(preProc, testExpCont[names(testExpCont) != "LogSalePrice"])

#######################################################################
### Use Linear Regression
#######################################################################
# Using regular data
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
LMmodel <- train(LogSalePrice ~.,
                 method = "lm",
                 trControl = ctrl,
                 preProc = c("center", "scale"),
                 data = trainExpCont
)

LMpred.trainlog <- predict(LMmodel, newdata = data.frame(trainExpCont))
LMpred.train <- exp(LMpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LMpred.train)

LMpred.testlog <- predict(LMmodel, newdata = data.frame(testExpCont))
LMpred.test <- exp(LMpred.testlog)

Id <- testExpCont_Id
SalePrice <- LMpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM1.csv", row.names = FALSE)

######################
### Correlation
######################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
LMmodel <- train(LogSalePrice ~. # - NewSale - HasPool - StreetPave - HasGarage
                 + FireplaceQu*FireplacesSqr2
                 + GarageCars*GarageArea 
                 + LogGrLivArea*LogTotRmsAbvGrd 
                 + TotalBsmtSF*Log1stFlrSF 
                 + LotFrontage*LotAreaSqr 
                 + MSSubClass*Has2Flrs
                 + HouseStyle*Has2Flrs 
                 + LogGrLivArea*LogTotalBath 
                 + Foundation*BsmtCond 
                 + TotalBsmtBath*BsmtTotalFinSFSq 
                 + BsmtCond*BsmtHeight 
                 + GarageCars*OverallQualSqr 
                 + BsmtFinType1*TotalBsmtBath 
                 + YearBuilt*YrSoldAfterRemodel 
                 + OverallQualSqr*LogGrLivArea 
                 + KitchenQual*OverallQualSqr 
                 + BldgType*LotAreaSqr 
                 + GarageArea*OverallQualSqr 
                 + YearBuilt*OverallQualSqr 
                 + YearBuilt*LogTotalBath 
                 + ExterQual*OverallQualSqr 
                 + Foundation*BsmtHeight 
                 + OverallQualSqr*LogTotalBath 
                 + TotalBsmtSF*OverallQualSqr 
                 + TotalBsmtSF*BsmtTotalFinSFSq 
                 + YearBuilt*GarageCars 
                 + OverallQualSqr*YrSoldAfterRemodel 
                 + MSSubClass*HouseStyle 
                 + Has2Flrs*LogGrLivArea 
                 + MasVnrType*MasVnrAreaSqr 
                 + BsmtFinType1*BsmtTotalFinSFSq 
                 + HeatingQC*YrSoldAfterRemodel 
                 + TotalBsmtSF*BsmtHeight 
                 + Log1stFlrSF*LogGrLivArea 
                 + YearBuilt*GarageFinish  
                 + ExterQual*KitchenQual 
                 + GarageCars*LogTotalBath 
                 + LotFrontage*BldgType
                 ##################################
                 + BsmtFinType1*BsmtUnfSF
                 + GarageCars*LogGrLivArea
                 + YearBuilt*GarageArea
                 + HouseStyle*LogGrLivArea
                 + LogTotalBath*LogTotRmsAbvGrd
                 + GarageFinish*OverallQualSqr
                 + YrSoldAfterRemodel*LogTotalBath
                 + Has2Flrs*LogTotRmsAbvGrd
                 + GarageArea*LogTotalBath
                 + LogGrLivArea*FireplacesSqr2
                 + GarageArea*Log1stFlrSF
                 + TotalBsmtSF*GarageArea
                 + GarageArea*LogGrLivArea
                 + BsmtTotalFinSFSq*Log1stFlrSF
                 + YearBuilt*Foundation
                 + BsmtUnfSF*TotalBsmtSF
                 + OverallQualSqr*Log1stFlrSF
                 + TotalBsmtBath*LogTotalBath
                 + BsmtUnfSF*TotalBsmtBath
                 + Foundation*TotalBsmtSF
                 + TotalBsmtSF*GarageCars
                 + HeatingQC*OverallQualSqr
                 + MSZoning*YearBuilt
                 + LotAreaSqr*Log1stFlrSF
                 + MSSubClass*LogGrLivArea
                 + GarageCars*Log1stFlrSF
                 + HouseStyle*LogTotRmsAbvGrd
                 + GarageFinish*LogTotalBath
                 + FireplaceQu*LogGrLivArea
                 + OverallQualSqr*MasVnrAreaSqr
                 + BsmtUnfSF*BsmtTotalFinSFSq
                 + OverallQualSqr*FireplacesSqr2
                 + GarageFinish*YrSoldAfterRemodel
                 + GarageCars*YrSoldAfterRemodel
                 + OverallQualSqr*BsmtHeight
                 + LotFrontage*Log1stFlrSF
                 + OverallQualSqr*LogTotRmsAbvGrd
                 + YearBuilt*OverallCondHigh
                 + YearBuilt*Remodeled
                 + MasVnrType*OverallQualSqr
                 + Neighborhood*OverallQualSqr
                 + BsmtExposure*BsmtTotalFinSFSq
                 + YearBuilt*HeatingQC
                 + YearBuilt*TotalBsmtSF
                 + Log1stFlrSF*LogTotRmsAbvGrd
                 + TotalBsmtSF*LogGrLivArea
                 ,
                 method = "lm",
                 trControl = ctrl,
                 preProc = c("center", "scale"),
                 data = trainExpCont
)

LMpred.trainlog <- predict(LMmodel, newdata = data.frame(trainExpCont))
LMpred.train <- exp(LMpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LMpred.train)
# 0.08867525

LMpred.testlog <- predict(LMmodel, newdata = data.frame(testExpCont))
LMpred.test <- exp(LMpred.testlog)

Id <- testExpCont_Id
SalePrice <- LMpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_Corr2.csv", row.names = FALSE)

######################
# Using PCA regression
######################
# set.seed(1)
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# LM_PCAmodel <- train(LogSalePrice ~.,
#                  method = "lm",
#                  trControl = ctrl,
#                  preProc = c("center", "scale"),
#                  data = trainPCA
# )
# 
# LM_PCApred.trainlog <- predict(LM_PCAmodel, newdata = data.frame(trainPCA))
# LM_PCApred.train <- exp(LM_PCApred.trainlog)
# myRMSLE(actual = trainExpCont_SalePrice, predicted = LM_PCApred.train)
# 
# LM_PCApred.testlog <- predict(LM_PCAmodel, newdata = data.frame(testPCA))
# LM_PCApred.test <- exp(LM_PCApred.testlog)
# 
# Id <- testExpCont_Id
# SalePrice <- LM_PCApred.test
# output <- data.frame(Id, SalePrice)
# write.csv(output, file = "housingSimpleLM_PCA.csv", row.names = FALSE)

#######################################################################
### Use Gradient Boosting Model
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
GBMmodel <- train(LogSalePrice ~.,
                  method = "gbm",
                  trControl = ctrl,
                  preProc = c("center", "scale"),
                  data = trainExpCont
)

GBMpred.trainlog <- predict(GBMmodel, newdata = data.frame(trainExpCont))
GBMpred.train <- exp(GBMpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = GBMpred.train)

GBMpred.testlog <- predict(GBMmodel, newdata = data.frame(testExpCont))
GBMpred.test <- exp(GBMpred.testlog)

Id <- testExpCont_Id
SalePrice <- GBMpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2GBM1.csv", row.names = FALSE)

######################
### Correlation
######################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
tune <- expand.grid(n.trees = seq(150, 450, by = 100), interaction.depth = c(1, 2, 3),
                    n.minobsinnode = 10, shrinkage = c(0.1, 0.2))
GBMmodel <- train(LogSalePrice ~. # - NewSale - HasPool - StreetPave - HasGarage
                  + FireplaceQu*FireplacesSqr2
                  + GarageCars*GarageArea 
                  + LogGrLivArea*LogTotRmsAbvGrd 
                  + TotalBsmtSF*Log1stFlrSF 
                  + LotFrontage*LotAreaSqr 
                  + MSSubClass*Has2Flrs
                  + HouseStyle*Has2Flrs 
                  + LogGrLivArea*LogTotalBath 
                  + Foundation*BsmtCond 
                  + TotalBsmtBath*BsmtTotalFinSFSq 
                  + BsmtCond*BsmtHeight 
                  + GarageCars*OverallQualSqr 
                  + BsmtFinType1*TotalBsmtBath 
                  + YearBuilt*YrSoldAfterRemodel 
                  + OverallQualSqr*LogGrLivArea 
                  + KitchenQual*OverallQualSqr 
                  + BldgType*LotAreaSqr 
                  + GarageArea*OverallQualSqr 
                  + YearBuilt*OverallQualSqr 
                  + YearBuilt*LogTotalBath 
                  + ExterQual*OverallQualSqr 
                  + Foundation*BsmtHeight 
                  + OverallQualSqr*LogTotalBath 
                  + TotalBsmtSF*OverallQualSqr 
                  + TotalBsmtSF*BsmtTotalFinSFSq 
                  + YearBuilt*GarageCars 
                  + OverallQualSqr*YrSoldAfterRemodel 
                  + MSSubClass*HouseStyle 
                  + Has2Flrs*LogGrLivArea 
                  + MasVnrType*MasVnrAreaSqr 
                  + BsmtFinType1*BsmtTotalFinSFSq 
                  + HeatingQC*YrSoldAfterRemodel 
                  + TotalBsmtSF*BsmtHeight 
                  + Log1stFlrSF*LogGrLivArea 
                  + YearBuilt*GarageFinish 
                  + ExterQual*KitchenQual 
                  + GarageCars*LogTotalBath 
                  + LotFrontage*BldgType
                  ##################################
                  + BsmtFinType1*BsmtUnfSF
                  + GarageCars*LogGrLivArea
                  + YearBuilt*GarageArea
                  + HouseStyle*LogGrLivArea
                  + LogTotalBath*LogTotRmsAbvGrd
                  + GarageFinish*OverallQualSqr
                  + YrSoldAfterRemodel*LogTotalBath
                  + Has2Flrs*LogTotRmsAbvGrd
                  + GarageArea*LogTotalBath
                  + LogGrLivArea*FireplacesSqr2
                  + GarageArea*Log1stFlrSF
                  + TotalBsmtSF*GarageArea
                  + GarageArea*LogGrLivArea
                  + BsmtTotalFinSFSq*Log1stFlrSF
                  + YearBuilt*Foundation
                  + BsmtUnfSF*TotalBsmtSF
                  + OverallQualSqr*Log1stFlrSF
                  + TotalBsmtBath*LogTotalBath
                  + BsmtUnfSF*TotalBsmtBath
                  + Foundation*TotalBsmtSF
                  + TotalBsmtSF*GarageCars
                  + HeatingQC*OverallQualSqr
                  + MSZoning*YearBuilt
                  + LotAreaSqr*Log1stFlrSF
                  + MSSubClass*LogGrLivArea
                  + GarageCars*Log1stFlrSF
                  + HouseStyle*LogTotRmsAbvGrd
                  + GarageFinish*LogTotalBath
                  + FireplaceQu*LogGrLivArea
                  + OverallQualSqr*MasVnrAreaSqr
                  + BsmtUnfSF*BsmtTotalFinSFSq
                  + OverallQualSqr*FireplacesSqr2
                  + GarageFinish*YrSoldAfterRemodel
                  + GarageCars*YrSoldAfterRemodel
                  + OverallQualSqr*BsmtHeight
                  + LotFrontage*Log1stFlrSF
                  + OverallQualSqr*LogTotRmsAbvGrd
                  + YearBuilt*OverallCondHigh
                  + YearBuilt*Remodeled
                  + MasVnrType*OverallQualSqr
                  + Neighborhood*OverallQualSqr
                  + BsmtExposure*BsmtTotalFinSFSq
                  + YearBuilt*HeatingQC
                  + YearBuilt*TotalBsmtSF
                  + Log1stFlrSF*LogTotRmsAbvGrd
                  + TotalBsmtSF*LogGrLivArea
                  ,
                  method = "gbm",
                  trControl = ctrl,
                  tuneGrid = tune,
                  preProc = c("center", "scale"),
                  data = trainExpCont
)

GBMpred.trainlog <- predict(GBMmodel, newdata = data.frame(trainExpCont))
GBMpred.train <- exp(GBMpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = GBMpred.train)
# 0.08477643 Fitting n.trees = 150, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 10 on full training set
# 0.06565379 Fitting n.trees = 350, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 10 on full training set

GBMpred.testlog <- predict(GBMmodel, newdata = data.frame(testExpCont))
GBMpred.test <- exp(GBMpred.testlog)

Id <- testExpCont_Id
SalePrice <- GBMpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2GBM_Corr2.csv", row.names = FALSE)

######################
# Using PCA regression
######################
# set.seed(1)
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# GBM_PCAmodel <- train(LogSalePrice ~.,
#                   method = "gbm",
#                   trControl = ctrl,
#                   preProc = c("center", "scale"),
#                   data = trainPCA
# )
# 
# GBM_PCApred.trainlog <- predict(GBM_PCAmodel, newdata = data.frame(trainPCA))
# GBM_PCApred.train <- exp(GBM_PCApred.trainlog)
# myRMSLE(actual = trainExpCont_SalePrice, predicted = GBM_PCApred.train)
# 
# GBM_PCApred.testlog <- predict(GBM_PCAmodel, newdata = data.frame(testPCA))
# GBM_PCApred.test <- exp(GBM_PCApred.testlog)
# 
# Id <- testExpCont_Id
# SalePrice <- GBM_PCApred.test
# output <- data.frame(Id, SalePrice)
# write.csv(output, file = "housingSimpleGBM_PCA.csv", row.names = FALSE)

#######################################################################
### Use Logrithmic averaging of LM and GBM
#######################################################################
LM_GBM_LogAvg.train <- exp((log(LMpred.train) + log(GBMpred.train)) / 2)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LM_GBM_LogAvg.train)
# 0.08737321

LM_GBM_LogAvg.test <- exp((log(LMpred.test) + log(GBMpred.test)) / 2)

Id <- testExpCont_Id
SalePrice <- LM_GBM_LogAvg.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_GBM1.csv", row.names = FALSE)

#################
### Correlation
#################

LM_GBM_LogAvg.train <- exp((log(LMpred.train) + log(GBMpred.train)) / 2)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LM_GBM_LogAvg.train)
# 0.08466397

LM_GBM_LogAvg.test <- exp((log(LMpred.test) + log(GBMpred.test)) / 2)

Id <- testExpCont_Id
SalePrice <- LM_GBM_LogAvg.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_GBM_Corr1.csv", row.names = FALSE)

#######################################################################
### Use xgbTree (Extreme Gradient Boosting using Trees)
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
XGBTmodel <- train(LogSalePrice ~.,
                   method = "xgbTree",
                   trControl = ctrl,
                   preProc = c("center", "scale"),
                   data = trainExpCont
)
# Fitting nrounds = 150, max_depth = 2, eta = 0.3, 
#         gamma = 0, colsample_bytree = 0.8, 
#         min_child_weight = 1, subsample = 1 on full training set

XGBTpred.trainlog <- predict(XGBTmodel, newdata = data.frame(trainExpCont))
XGBTpred.train <- exp(XGBTpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = XGBTpred.train)
# 0.08392772

XGBTpred.testlog <- predict(XGBTmodel, newdata = data.frame(testExpCont))
XGBTpred.test <- exp(XGBTpred.testlog)

Id <- testExpCont_Id
SalePrice <- XGBTpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2XGBT1.csv", row.names = FALSE)

######################
### Intensive xgbTree
######################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 1, number = 5, verboseIter = TRUE)
tune <- expand.grid(nrounds = 1000, 
                    max_depth = # c(1, 2), 
                      # c(1, 2, 3, 4), 
                      c(3, 4, 5, 6),
                    eta = seq(0.01, 0.1, by = 0.01), 
                    # seq(0.01, 0.15, by = 0.01),
                    gamma = c(0, 1, 2, 3),
                    # c(0, 1), 
                    # c(0, 1, 2, 3),
                    colsample_bytree = c(0.6, 0.7, 0.8), 
                    # c(0.5, 0.6, 0.7, 0.8),
                    subsample = c(0.5, 0.75, 1), 
                    # c(0.5, 0.6, 0.7),
                    min_child_weight = seq(1, 5, by = 1) 
                    # seq(0, 9, by = 1)
)

XGBTmodel <- train(LogSalePrice ~. # - NewSale - HasPool - StreetPave - HasGarage
                   + FireplaceQu*FireplacesSqr2
                   + GarageCars*GarageArea 
                   + LogGrLivArea*LogTotRmsAbvGrd 
                   + TotalBsmtSF*Log1stFlrSF 
                   + LotFrontage*LotAreaSqr 
                   + MSSubClass*Has2Flrs
                   + HouseStyle*Has2Flrs 
                   + LogGrLivArea*LogTotalBath 
                   + Foundation*BsmtCond 
                   + TotalBsmtBath*BsmtTotalFinSFSq 
                   + BsmtCond*BsmtHeight 
                   + GarageCars*OverallQualSqr 
                   + BsmtFinType1*TotalBsmtBath 
                   + YearBuilt*YrSoldAfterRemodel 
                   + OverallQualSqr*LogGrLivArea 
                   + KitchenQual*OverallQualSqr 
                   + BldgType*LotAreaSqr 
                   + GarageArea*OverallQualSqr 
                   + YearBuilt*OverallQualSqr 
                   + YearBuilt*LogTotalBath 
                   + ExterQual*OverallQualSqr 
                   + Foundation*BsmtHeight 
                   + OverallQualSqr*LogTotalBath 
                   + TotalBsmtSF*OverallQualSqr 
                   + TotalBsmtSF*BsmtTotalFinSFSq 
                   + YearBuilt*GarageCars 
                   + OverallQualSqr*YrSoldAfterRemodel 
                   + MSSubClass*HouseStyle 
                   + Has2Flrs*LogGrLivArea 
                   + MasVnrType*MasVnrAreaSqr 
                   + BsmtFinType1*BsmtTotalFinSFSq 
                   + HeatingQC*YrSoldAfterRemodel 
                   + TotalBsmtSF*BsmtHeight 
                   + Log1stFlrSF*LogGrLivArea 
                   + YearBuilt*GarageFinish 
                   + ExterQual*KitchenQual 
                   + GarageCars*LogTotalBath 
                   + LotFrontage*BldgType
                   ,
                   method = "xgbTree",
                   trControl = ctrl,
                   tuneGrid = tune,
                   verbose = TRUE,
                   preProc = c("center", "scale"),
                   data = trainExpCont
)

XGBTpred.trainlog <- predict(XGBTmodel, newdata = data.frame(trainExpCont))
XGBTpred.train <- exp(XGBTpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = XGBTpred.train)

# Fitting nrounds = 500, max_depth = 3, eta = 0.06, 
#         gamma = 0, colsample_bytree = 0.5, 
#         min_child_weight = 2, subsample = 0.5 on full training set
# 0.06544136
# housingSimple2XGBT_Corr1.csv

XGBTpred.testlog <- predict(XGBTmodel, newdata = data.frame(testExpCont))
XGBTpred.test <- exp(XGBTpred.testlog)

Id <- testExpCont_Id
SalePrice <- XGBTpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2XGBT_Corr2.csv", row.names = FALSE)

#######################################################################
### Use xgbLinear (Extreme Gradient Boosting using Linearization)
#######################################################################
# set.seed(1)
# ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
# XGBLmodel <- train(LogSalePrice ~.,
#                    method = "xgbLinear",
#                    trControl = ctrl,
#                    preProc = c("center", "scale"),
#                    data = trainExpCont
# )
# 
# XGBLpred.trainlog <- predict(XGBLmodel, newdata = data.frame(trainExpCont))
# XGBLpred.train <- exp(XGBLpred.trainlog)
# myRMSLE(actual = trainExpCont_SalePrice, predicted = XGBLpred.train)
# # 0.02634121
# 
# XGBLpred.testlog <- predict(XGBLmodel, newdata = data.frame(testExpCont))
# XGBLpred.test <- exp(XGBLpred.testlog)
# 
# Id <- testExpCont_Id
# SalePrice <- XGBLpred.test
# output <- data.frame(Id, SalePrice)
# write.csv(output, file = "housingSimple2XGBL1.csv", row.names = FALSE)

#########################
### Intensive xgbLinear
#########################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 1, number = 5, verboseIter = TRUE)
tune <- expand.grid(nrounds = 2000, 
                    eta = seq(0.0005, 0.01, by = 0.0005),
                    alpha = seq(0.7, 0.9, by = 0.1),
                    lambda = seq(0.4, 0.6, by = 0.1)
)
XGBLmodel <- train(LogSalePrice ~. # - NewSale - HasPool - StreetPave - HasGarage
                   + FireplaceQu*FireplacesSqr2
                   + GarageCars*GarageArea 
                   + LogGrLivArea*LogTotRmsAbvGrd 
                   + TotalBsmtSF*Log1stFlrSF 
                   + LotFrontage*LotAreaSqr 
                   + MSSubClass*Has2Flrs
                   + HouseStyle*Has2Flrs 
                   + LogGrLivArea*LogTotalBath 
                   + Foundation*BsmtCond 
                   + TotalBsmtBath*BsmtTotalFinSFSq 
                   + BsmtCond*BsmtHeight 
                   + GarageCars*OverallQualSqr 
                   + BsmtFinType1*TotalBsmtBath 
                   + YearBuilt*YrSoldAfterRemodel 
                   + OverallQualSqr*LogGrLivArea 
                   + KitchenQual*OverallQualSqr 
                   + BldgType*LotAreaSqr 
                   + GarageArea*OverallQualSqr 
                   + YearBuilt*OverallQualSqr 
                   + YearBuilt*LogTotalBath 
                   + ExterQual*OverallQualSqr 
                   + Foundation*BsmtHeight 
                   + OverallQualSqr*LogTotalBath 
                   + TotalBsmtSF*OverallQualSqr 
                   + TotalBsmtSF*BsmtTotalFinSFSq 
                   + YearBuilt*GarageCars 
                   + OverallQualSqr*YrSoldAfterRemodel 
                   + MSSubClass*HouseStyle 
                   + Has2Flrs*LogGrLivArea 
                   + MasVnrType*MasVnrAreaSqr 
                   + BsmtFinType1*BsmtTotalFinSFSq 
                   + HeatingQC*YrSoldAfterRemodel 
                   + TotalBsmtSF*BsmtHeight 
                   + Log1stFlrSF*LogGrLivArea 
                   + YearBuilt*GarageFinish 
                   + ExterQual*KitchenQual 
                   + GarageCars*LogTotalBath 
                   + LotFrontage*BldgType
                   ,
                   method = "xgbLinear",
                   trControl = ctrl,
                   tuneGrid = tune,
                   verbose = TRUE,
                   preProc = c("center", "scale"),
                   data = trainExpCont
)

XGBLpred.trainlog <- predict(XGBLmodel, newdata = data.frame(trainExpCont))
XGBLpred.train <- exp(XGBLpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = XGBLpred.train)

# Fitting nrounds = 500, lambda = 0.5, alpha = 0.8, eta = 0.01 on full training set
# 0.04359939 housingSimple2XGBL_Corr1.csv

# Fitting nrounds = 1000, lambda = 0.8, alpha = 0.4, eta = 0.001 on full training set
# 0.0251251 housingSimple2XGBL_Corr2.csv

# Fitting nrounds = 2000, lambda = 0.5, alpha = 0.8, eta = 5e-04 on full training set
# 0.04359939 housingSimple2XGBL_Corr3.csv

XGBLpred.testlog <- predict(XGBLmodel, newdata = data.frame(testExpCont))
XGBLpred.test <- exp(XGBLpred.testlog)

Id <- testExpCont_Id
SalePrice <- XGBLpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2XGBL_Corr3.csv", row.names = FALSE)

#######################################################################
### Use ENET: ElasticNet, between Lasso and Ridge
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
ENETmodel <- train(LogSalePrice ~.,
                   method = "enet",
                   trControl = ctrl,
                   preProc = c("center", "scale"),
                   data = trainExpCont
)

ENETpred.trainlog <- predict(ENETmodel, newdata = data.frame(trainExpCont))
ENETpred.train <- exp(ENETpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = ENETpred.train)
# 0.09372048

ENETpred.testlog <- predict(ENETmodel, newdata = data.frame(testExpCont))
ENETpred.test <- exp(ENETpred.testlog)

Id <- testExpCont_Id
SalePrice <- ENETpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2ENET1.csv", row.names = FALSE)

##################
### Correlation
##################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
ENETmodel <- train(LogSalePrice ~. - StreetPave - HasPool # - NewSale - HasPool - StreetPave - HasGarage
                   + FireplaceQu*FireplacesSqr2
                   + GarageCars*GarageArea 
                   + LogGrLivArea*LogTotRmsAbvGrd 
                   + TotalBsmtSF*Log1stFlrSF 
                   + LotFrontage*LotAreaSqr 
                   + MSSubClass*Has2Flrs
                   + HouseStyle*Has2Flrs 
                   + LogGrLivArea*LogTotalBath 
                   + Foundation*BsmtCond 
                   + TotalBsmtBath*BsmtTotalFinSFSq 
                   + BsmtCond*BsmtHeight 
                   + GarageCars*OverallQualSqr 
                   + BsmtFinType1*TotalBsmtBath 
                   + YearBuilt*YrSoldAfterRemodel 
                   + OverallQualSqr*LogGrLivArea 
                   + KitchenQual*OverallQualSqr 
                   + BldgType*LotAreaSqr 
                   + GarageArea*OverallQualSqr 
                   + YearBuilt*OverallQualSqr 
                   + YearBuilt*LogTotalBath 
                   + ExterQual*OverallQualSqr 
                   + Foundation*BsmtHeight 
                   + OverallQualSqr*LogTotalBath 
                   + TotalBsmtSF*OverallQualSqr 
                   + TotalBsmtSF*BsmtTotalFinSFSq 
                   + YearBuilt*GarageCars 
                   + OverallQualSqr*YrSoldAfterRemodel 
                   + MSSubClass*HouseStyle 
                   + Has2Flrs*LogGrLivArea 
                   + MasVnrType*MasVnrAreaSqr 
                   + BsmtFinType1*BsmtTotalFinSFSq 
                   + HeatingQC*YrSoldAfterRemodel 
                   + TotalBsmtSF*BsmtHeight 
                   + Log1stFlrSF*LogGrLivArea 
                   + YearBuilt*GarageFinish 
                   + ExterQual*KitchenQual 
                   + GarageCars*LogTotalBath 
                   + LotFrontage*BldgType
                   ##################################
                   + BsmtFinType1*BsmtUnfSF
                   + GarageCars*LogGrLivArea
                   + YearBuilt*GarageArea
                   + HouseStyle*LogGrLivArea
                   + LogTotalBath*LogTotRmsAbvGrd
                   + GarageFinish*OverallQualSqr
                   + YrSoldAfterRemodel*LogTotalBath
                   + Has2Flrs*LogTotRmsAbvGrd
                   + GarageArea*LogTotalBath
                   + LogGrLivArea*FireplacesSqr2
                   + GarageArea*Log1stFlrSF
                   + TotalBsmtSF*GarageArea
                   + GarageArea*LogGrLivArea
                   + BsmtTotalFinSFSq*Log1stFlrSF
                   + YearBuilt*Foundation
                   + BsmtUnfSF*TotalBsmtSF
                   + OverallQualSqr*Log1stFlrSF
                   + TotalBsmtBath*LogTotalBath
                   + BsmtUnfSF*TotalBsmtBath
                   + Foundation*TotalBsmtSF
                   + TotalBsmtSF*GarageCars
                   + HeatingQC*OverallQualSqr
                   + MSZoning*YearBuilt
                   + LotAreaSqr*Log1stFlrSF
                   + MSSubClass*LogGrLivArea
                   + GarageCars*Log1stFlrSF
                   + HouseStyle*LogTotRmsAbvGrd
                   + GarageFinish*LogTotalBath
                   + FireplaceQu*LogGrLivArea
                   + OverallQualSqr*MasVnrAreaSqr
                   + BsmtUnfSF*BsmtTotalFinSFSq
                   + OverallQualSqr*FireplacesSqr2
                   + GarageFinish*YrSoldAfterRemodel
                   + GarageCars*YrSoldAfterRemodel
                   + OverallQualSqr*BsmtHeight
                   + LotFrontage*Log1stFlrSF
                   + OverallQualSqr*LogTotRmsAbvGrd
                   + YearBuilt*OverallCondHigh
                   + YearBuilt*Remodeled
                   + MasVnrType*OverallQualSqr
                   + Neighborhood*OverallQualSqr
                   + BsmtExposure*BsmtTotalFinSFSq
                   + YearBuilt*HeatingQC
                   + YearBuilt*TotalBsmtSF
                   + Log1stFlrSF*LogTotRmsAbvGrd
                   + TotalBsmtSF*LogGrLivArea
                   ,
                   method = "enet",
                   trControl = ctrl,
                   preProc = c("center", "scale"),
                   data = trainExpCont
)

ENETpred.trainlog <- predict(ENETmodel, newdata = data.frame(trainExpCont))
ENETpred.train <- exp(ENETpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = ENETpred.train)
# 0.0928416 Fitting fraction = 0.05, lambda = 0 on full training set

ENETpred.testlog <- predict(ENETmodel, newdata = data.frame(testExpCont))
ENETpred.test <- exp(ENETpred.testlog)

Id <- testExpCont_Id
SalePrice <- ENETpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2ENET_Corr2.csv", row.names = FALSE)

#######################################################################
### Use Lasso Regression
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
tune <- expand.grid(fraction = seq(0.1, 0.9, by = 0.1))
LASSOmodel <- train(LogSalePrice ~. # - StreetPave - HasPool # - NewSale - HasPool - StreetPave - HasGarage
                    + FireplaceQu*FireplacesSqr2
                    + GarageCars*GarageArea 
                    + LogGrLivArea*LogTotRmsAbvGrd 
                    + TotalBsmtSF*Log1stFlrSF 
                    + LotFrontage*LotAreaSqr 
                    + MSSubClass*Has2Flrs
                    + HouseStyle*Has2Flrs 
                    + LogGrLivArea*LogTotalBath 
                    + Foundation*BsmtCond 
                    + TotalBsmtBath*BsmtTotalFinSFSq 
                    + BsmtCond*BsmtHeight 
                    + GarageCars*OverallQualSqr 
                    + BsmtFinType1*TotalBsmtBath 
                    + YearBuilt*YrSoldAfterRemodel 
                    + OverallQualSqr*LogGrLivArea 
                    + KitchenQual*OverallQualSqr 
                    + BldgType*LotAreaSqr 
                    + GarageArea*OverallQualSqr 
                    + YearBuilt*OverallQualSqr 
                    + YearBuilt*LogTotalBath 
                    + ExterQual*OverallQualSqr 
                    + Foundation*BsmtHeight 
                    + OverallQualSqr*LogTotalBath 
                    + TotalBsmtSF*OverallQualSqr 
                    + TotalBsmtSF*BsmtTotalFinSFSq 
                    + YearBuilt*GarageCars 
                    + OverallQualSqr*YrSoldAfterRemodel 
                    + MSSubClass*HouseStyle 
                    + Has2Flrs*LogGrLivArea 
                    + MasVnrType*MasVnrAreaSqr 
                    + BsmtFinType1*BsmtTotalFinSFSq 
                    + HeatingQC*YrSoldAfterRemodel 
                    + TotalBsmtSF*BsmtHeight 
                    + Log1stFlrSF*LogGrLivArea 
                    + YearBuilt*GarageFinish 
                    + ExterQual*KitchenQual 
                    + GarageCars*LogTotalBath 
                    + LotFrontage*BldgType
                    ##################################
                    + BsmtFinType1*BsmtUnfSF
                    + GarageCars*LogGrLivArea
                    + YearBuilt*GarageArea
                    + HouseStyle*LogGrLivArea
                    + LogTotalBath*LogTotRmsAbvGrd
                    + GarageFinish*OverallQualSqr
                    + YrSoldAfterRemodel*LogTotalBath
                    + Has2Flrs*LogTotRmsAbvGrd
                    + GarageArea*LogTotalBath
                    + LogGrLivArea*FireplacesSqr2
                    + GarageArea*Log1stFlrSF
                    + TotalBsmtSF*GarageArea
                    + GarageArea*LogGrLivArea
                    + BsmtTotalFinSFSq*Log1stFlrSF
                    + YearBuilt*Foundation
                    + BsmtUnfSF*TotalBsmtSF
                    + OverallQualSqr*Log1stFlrSF
                    + TotalBsmtBath*LogTotalBath
                    + BsmtUnfSF*TotalBsmtBath
                    + Foundation*TotalBsmtSF
                    + TotalBsmtSF*GarageCars
                    + HeatingQC*OverallQualSqr
                    + MSZoning*YearBuilt
                    + LotAreaSqr*Log1stFlrSF
                    + MSSubClass*LogGrLivArea
                    + GarageCars*Log1stFlrSF
                    + HouseStyle*LogTotRmsAbvGrd
                    + GarageFinish*LogTotalBath
                    + FireplaceQu*LogGrLivArea
                    + OverallQualSqr*MasVnrAreaSqr
                    + BsmtUnfSF*BsmtTotalFinSFSq
                    + OverallQualSqr*FireplacesSqr2
                    + GarageFinish*YrSoldAfterRemodel
                    + GarageCars*YrSoldAfterRemodel
                    + OverallQualSqr*BsmtHeight
                    + LotFrontage*Log1stFlrSF
                    + OverallQualSqr*LogTotRmsAbvGrd
                    + YearBuilt*OverallCondHigh
                    + YearBuilt*Remodeled
                    + MasVnrType*OverallQualSqr
                    + Neighborhood*OverallQualSqr
                    + BsmtExposure*BsmtTotalFinSFSq
                    + YearBuilt*HeatingQC
                    + YearBuilt*TotalBsmtSF
                    + Log1stFlrSF*LogTotRmsAbvGrd
                    + TotalBsmtSF*LogGrLivArea
                    ,
                    method = "lasso",
                    trControl = ctrl,
                    tuneGrid = tune,
                    preProc = c("center", "scale"),
                    data = trainExpCont
)

LASSOpred.trainlog <- predict(LASSOmodel, newdata = data.frame(trainExpCont))
LASSOpred.train <- exp(LASSOpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LASSOpred.train)
# 0.0927618 Fitting fraction = 0.1 on full training set
# 0.09107312 Fitting fraction = 0.1 on full training set

LASSOpred.testlog <- predict(LASSOmodel, newdata = data.frame(testExpCont))
LASSOpred.test <- exp(LASSOpred.testlog)

Id <- testExpCont_Id
SalePrice <- LASSOpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LASSO_Corr1.csv", row.names = FALSE)

#######################################################################
### Use Ridge Regression
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
RIDGEmodel <- train(LogSalePrice ~. # - StreetPave - HasPool # - NewSale - HasPool - StreetPave - HasGarage
                    + FireplaceQu*FireplacesSqr2
                    + GarageCars*GarageArea 
                    + LogGrLivArea*LogTotRmsAbvGrd 
                    + TotalBsmtSF*Log1stFlrSF 
                    + LotFrontage*LotAreaSqr 
                    + MSSubClass*Has2Flrs
                    + HouseStyle*Has2Flrs 
                    + LogGrLivArea*LogTotalBath 
                    + Foundation*BsmtCond 
                    + TotalBsmtBath*BsmtTotalFinSFSq 
                    + BsmtCond*BsmtHeight 
                    + GarageCars*OverallQualSqr 
                    + BsmtFinType1*TotalBsmtBath 
                    + YearBuilt*YrSoldAfterRemodel 
                    + OverallQualSqr*LogGrLivArea 
                    + KitchenQual*OverallQualSqr 
                    + BldgType*LotAreaSqr 
                    + GarageArea*OverallQualSqr 
                    + YearBuilt*OverallQualSqr 
                    + YearBuilt*LogTotalBath 
                    + ExterQual*OverallQualSqr 
                    + Foundation*BsmtHeight 
                    + OverallQualSqr*LogTotalBath 
                    + TotalBsmtSF*OverallQualSqr 
                    + TotalBsmtSF*BsmtTotalFinSFSq 
                    + YearBuilt*GarageCars 
                    + OverallQualSqr*YrSoldAfterRemodel 
                    + MSSubClass*HouseStyle 
                    + Has2Flrs*LogGrLivArea 
                    + MasVnrType*MasVnrAreaSqr 
                    + BsmtFinType1*BsmtTotalFinSFSq 
                    + HeatingQC*YrSoldAfterRemodel 
                    + TotalBsmtSF*BsmtHeight 
                    + Log1stFlrSF*LogGrLivArea 
                    + YearBuilt*GarageFinish 
                    + ExterQual*KitchenQual 
                    + GarageCars*LogTotalBath 
                    + LotFrontage*BldgType
                    ##################################
                    + BsmtFinType1*BsmtUnfSF
                    + GarageCars*LogGrLivArea
                    + YearBuilt*GarageArea
                    + HouseStyle*LogGrLivArea
                    + LogTotalBath*LogTotRmsAbvGrd
                    + GarageFinish*OverallQualSqr
                    + YrSoldAfterRemodel*LogTotalBath
                    + Has2Flrs*LogTotRmsAbvGrd
                    + GarageArea*LogTotalBath
                    + LogGrLivArea*FireplacesSqr2
                    + GarageArea*Log1stFlrSF
                    + TotalBsmtSF*GarageArea
                    + GarageArea*LogGrLivArea
                    + BsmtTotalFinSFSq*Log1stFlrSF
                    + YearBuilt*Foundation
                    + BsmtUnfSF*TotalBsmtSF
                    + OverallQualSqr*Log1stFlrSF
                    + TotalBsmtBath*LogTotalBath
                    + BsmtUnfSF*TotalBsmtBath
                    + Foundation*TotalBsmtSF
                    + TotalBsmtSF*GarageCars
                    + HeatingQC*OverallQualSqr
                    + MSZoning*YearBuilt
                    + LotAreaSqr*Log1stFlrSF
                    + MSSubClass*LogGrLivArea
                    + GarageCars*Log1stFlrSF
                    + HouseStyle*LogTotRmsAbvGrd
                    + GarageFinish*LogTotalBath
                    + FireplaceQu*LogGrLivArea
                    + OverallQualSqr*MasVnrAreaSqr
                    + BsmtUnfSF*BsmtTotalFinSFSq
                    + OverallQualSqr*FireplacesSqr2
                    + GarageFinish*YrSoldAfterRemodel
                    + GarageCars*YrSoldAfterRemodel
                    + OverallQualSqr*BsmtHeight
                    + LotFrontage*Log1stFlrSF
                    + OverallQualSqr*LogTotRmsAbvGrd
                    + YearBuilt*OverallCondHigh
                    + YearBuilt*Remodeled
                    + MasVnrType*OverallQualSqr
                    + Neighborhood*OverallQualSqr
                    + BsmtExposure*BsmtTotalFinSFSq
                    + YearBuilt*HeatingQC
                    + YearBuilt*TotalBsmtSF
                    + Log1stFlrSF*LogTotRmsAbvGrd
                    + TotalBsmtSF*LogGrLivArea
                    ,
                    method = "ridge",
                    trControl = ctrl,
                    preProc = c("center", "scale"),
                    data = trainExpCont
)

RIDGEpred.trainlog <- predict(RIDGEmodel, newdata = data.frame(trainExpCont))
RIDGEpred.train <- exp(RIDGEpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = RIDGEpred.train)
# 0.09114071 Fitting lambda = 1e-04 on full training set
# 0.08947825 Fitting lambda = 1e-04 on full training set

RIDGEpred.testlog <- predict(RIDGEmodel, newdata = data.frame(testExpCont))
RIDGEpred.test <- exp(RIDGEpred.testlog)

Id <- testExpCont_Id
SalePrice <- RIDGEpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2RIDGE_Corr1.csv", row.names = FALSE)

#######################################################################
### Use glmnet: Lasso and Elastic-Net Regularized Generalized Linear Models
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
tune <- expand.grid(alpha = seq(0, 1, by = 0.1), lambda = 0.000663)
GLMNETmodel <- train(LogSalePrice ~. # - StreetPave - HasPool # - NewSale - HasPool - StreetPave - HasGarage
                     + FireplaceQu*FireplacesSqr2
                     + GarageCars*GarageArea 
                     + LogGrLivArea*LogTotRmsAbvGrd 
                     + TotalBsmtSF*Log1stFlrSF 
                     + LotFrontage*LotAreaSqr 
                     + MSSubClass*Has2Flrs
                     + HouseStyle*Has2Flrs 
                     + LogGrLivArea*LogTotalBath 
                     + Foundation*BsmtCond 
                     + TotalBsmtBath*BsmtTotalFinSFSq 
                     + BsmtCond*BsmtHeight 
                     + GarageCars*OverallQualSqr 
                     + BsmtFinType1*TotalBsmtBath 
                     + YearBuilt*YrSoldAfterRemodel 
                     + OverallQualSqr*LogGrLivArea 
                     + KitchenQual*OverallQualSqr 
                     + BldgType*LotAreaSqr 
                     + GarageArea*OverallQualSqr 
                     + YearBuilt*OverallQualSqr 
                     + YearBuilt*LogTotalBath 
                     + ExterQual*OverallQualSqr 
                     + Foundation*BsmtHeight 
                     + OverallQualSqr*LogTotalBath 
                     + TotalBsmtSF*OverallQualSqr 
                     + TotalBsmtSF*BsmtTotalFinSFSq 
                     + YearBuilt*GarageCars 
                     + OverallQualSqr*YrSoldAfterRemodel 
                     + MSSubClass*HouseStyle 
                     + Has2Flrs*LogGrLivArea 
                     + MasVnrType*MasVnrAreaSqr 
                     + BsmtFinType1*BsmtTotalFinSFSq 
                     + HeatingQC*YrSoldAfterRemodel 
                     + TotalBsmtSF*BsmtHeight 
                     + Log1stFlrSF*LogGrLivArea 
                     + YearBuilt*GarageFinish 
                     + ExterQual*KitchenQual 
                     + GarageCars*LogTotalBath 
                     + LotFrontage*BldgType
                     ##################################
                     + BsmtFinType1*BsmtUnfSF
                     + GarageCars*LogGrLivArea
                     + YearBuilt*GarageArea
                     + HouseStyle*LogGrLivArea
                     + LogTotalBath*LogTotRmsAbvGrd
                     + GarageFinish*OverallQualSqr
                     + YrSoldAfterRemodel*LogTotalBath
                     + Has2Flrs*LogTotRmsAbvGrd
                     + GarageArea*LogTotalBath
                     + LogGrLivArea*FireplacesSqr2
                     + GarageArea*Log1stFlrSF
                     + TotalBsmtSF*GarageArea
                     + GarageArea*LogGrLivArea
                     + BsmtTotalFinSFSq*Log1stFlrSF
                     + YearBuilt*Foundation
                     + BsmtUnfSF*TotalBsmtSF
                     + OverallQualSqr*Log1stFlrSF
                     + TotalBsmtBath*LogTotalBath
                     + BsmtUnfSF*TotalBsmtBath
                     + Foundation*TotalBsmtSF
                     + TotalBsmtSF*GarageCars
                     + HeatingQC*OverallQualSqr
                     + MSZoning*YearBuilt
                     + LotAreaSqr*Log1stFlrSF
                     + MSSubClass*LogGrLivArea
                     + GarageCars*Log1stFlrSF
                     + HouseStyle*LogTotRmsAbvGrd
                     + GarageFinish*LogTotalBath
                     + FireplaceQu*LogGrLivArea
                     + OverallQualSqr*MasVnrAreaSqr
                     + BsmtUnfSF*BsmtTotalFinSFSq
                     + OverallQualSqr*FireplacesSqr2
                     + GarageFinish*YrSoldAfterRemodel
                     + GarageCars*YrSoldAfterRemodel
                     + OverallQualSqr*BsmtHeight
                     + LotFrontage*Log1stFlrSF
                     + OverallQualSqr*LogTotRmsAbvGrd
                     + YearBuilt*OverallCondHigh
                     + YearBuilt*Remodeled
                     + MasVnrType*OverallQualSqr
                     + Neighborhood*OverallQualSqr
                     + BsmtExposure*BsmtTotalFinSFSq
                     + YearBuilt*HeatingQC
                     + YearBuilt*TotalBsmtSF
                     + Log1stFlrSF*LogTotRmsAbvGrd
                     + TotalBsmtSF*LogGrLivArea
                     ,
                     method = "glmnet",
                     trControl = ctrl,
                     tuneGrid = tune,
                     preProc = c("center", "scale"),
                     data = trainExpCont
)

GLMNETpred.trainlog <- predict(GLMNETmodel, newdata = data.frame(trainExpCont))
GLMNETpred.train <- exp(GLMNETpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = GLMNETpred.train)
# 0.09334619 Fitting alpha = 1, lambda = 0.000663 on full training set
# 0.09324055 Fitting alpha = 1, lambda = 0.001 on full training set
# 0.09301005 Fitting alpha = 1, lambda = 0.000663 on full training set

GLMNETpred.testlog <- predict(GLMNETmodel, newdata = data.frame(testExpCont))
GLMNETpred.test <- exp(GLMNETpred.testlog)

Id <- testExpCont_Id
SalePrice <- GLMNETpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2GLMNET_Corr1.csv", row.names = FALSE)

#######################################################################
### Use bayesglm: Bayesian Generalized Linear Model
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
BAYESGLMmodel <- train(LogSalePrice ~. # - StreetPave - HasPool # - NewSale - HasPool - StreetPave - HasGarage
                       + FireplaceQu*FireplacesSqr2
                       + GarageCars*GarageArea 
                       + LogGrLivArea*LogTotRmsAbvGrd 
                       + TotalBsmtSF*Log1stFlrSF 
                       + LotFrontage*LotAreaSqr 
                       + MSSubClass*Has2Flrs
                       + HouseStyle*Has2Flrs 
                       + LogGrLivArea*LogTotalBath 
                       + Foundation*BsmtCond 
                       + TotalBsmtBath*BsmtTotalFinSFSq 
                       + BsmtCond*BsmtHeight 
                       + GarageCars*OverallQualSqr 
                       + BsmtFinType1*TotalBsmtBath 
                       + YearBuilt*YrSoldAfterRemodel 
                       + OverallQualSqr*LogGrLivArea 
                       + KitchenQual*OverallQualSqr 
                       + BldgType*LotAreaSqr 
                       + GarageArea*OverallQualSqr 
                       + YearBuilt*OverallQualSqr 
                       + YearBuilt*LogTotalBath 
                       + ExterQual*OverallQualSqr 
                       + Foundation*BsmtHeight 
                       + OverallQualSqr*LogTotalBath 
                       + TotalBsmtSF*OverallQualSqr 
                       + TotalBsmtSF*BsmtTotalFinSFSq 
                       + YearBuilt*GarageCars 
                       + OverallQualSqr*YrSoldAfterRemodel 
                       + MSSubClass*HouseStyle 
                       + Has2Flrs*LogGrLivArea 
                       + MasVnrType*MasVnrAreaSqr 
                       + BsmtFinType1*BsmtTotalFinSFSq 
                       + HeatingQC*YrSoldAfterRemodel 
                       + TotalBsmtSF*BsmtHeight 
                       + Log1stFlrSF*LogGrLivArea 
                       + YearBuilt*GarageFinish 
                       + ExterQual*KitchenQual 
                       + GarageCars*LogTotalBath 
                       + LotFrontage*BldgType
                       ##################################
                       + BsmtFinType1*BsmtUnfSF
                       + GarageCars*LogGrLivArea
                       + YearBuilt*GarageArea
                       + HouseStyle*LogGrLivArea
                       + LogTotalBath*LogTotRmsAbvGrd
                       + GarageFinish*OverallQualSqr
                       + YrSoldAfterRemodel*LogTotalBath
                       + Has2Flrs*LogTotRmsAbvGrd
                       + GarageArea*LogTotalBath
                       + LogGrLivArea*FireplacesSqr2
                       + GarageArea*Log1stFlrSF
                       + TotalBsmtSF*GarageArea
                       + GarageArea*LogGrLivArea
                       + BsmtTotalFinSFSq*Log1stFlrSF
                       + YearBuilt*Foundation
                       + BsmtUnfSF*TotalBsmtSF
                       + OverallQualSqr*Log1stFlrSF
                       + TotalBsmtBath*LogTotalBath
                       + BsmtUnfSF*TotalBsmtBath
                       + Foundation*TotalBsmtSF
                       + TotalBsmtSF*GarageCars
                       + HeatingQC*OverallQualSqr
                       + MSZoning*YearBuilt
                       + LotAreaSqr*Log1stFlrSF
                       + MSSubClass*LogGrLivArea
                       + GarageCars*Log1stFlrSF
                       + HouseStyle*LogTotRmsAbvGrd
                       + GarageFinish*LogTotalBath
                       + FireplaceQu*LogGrLivArea
                       + OverallQualSqr*MasVnrAreaSqr
                       + BsmtUnfSF*BsmtTotalFinSFSq
                       + OverallQualSqr*FireplacesSqr2
                       + GarageFinish*YrSoldAfterRemodel
                       + GarageCars*YrSoldAfterRemodel
                       + OverallQualSqr*BsmtHeight
                       + LotFrontage*Log1stFlrSF
                       + OverallQualSqr*LogTotRmsAbvGrd
                       + YearBuilt*OverallCondHigh
                       + YearBuilt*Remodeled
                       + MasVnrType*OverallQualSqr
                       + Neighborhood*OverallQualSqr
                       + BsmtExposure*BsmtTotalFinSFSq
                       + YearBuilt*HeatingQC
                       + YearBuilt*TotalBsmtSF
                       + Log1stFlrSF*LogTotRmsAbvGrd
                       + TotalBsmtSF*LogGrLivArea
                       ,
                       method = "bayesglm",
                       trControl = ctrl,
                       preProc = c("center", "scale"),
                       data = trainExpCont
)

BAYESGLMpred.trainlog <- predict(BAYESGLMmodel, newdata = data.frame(trainExpCont))
BAYESGLMpred.train <- exp(BAYESGLMpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = BAYESGLMpred.train)
# 0.09060447
# 0.08869086

BAYESGLMpred.testlog <- predict(BAYESGLMmodel, newdata = data.frame(testExpCont))
BAYESGLMpred.test <- exp(BAYESGLMpred.testlog)

Id <- testExpCont_Id
SalePrice <- BAYESGLMpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2BAYESGLM_Corr1.csv", row.names = FALSE)

#######################################################################
### Use glmboost: Boosted Generalized Linear Model
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
tune <- expand.grid(mstop = seq(50, 500, by = 50), prune = c('yes', 'no'))
GLMBOOSTmodel <- train(LogSalePrice ~. # - StreetPave - HasPool # - NewSale - HasPool - StreetPave - HasGarage
                       + FireplaceQu*FireplacesSqr2
                       + GarageCars*GarageArea 
                       + LogGrLivArea*LogTotRmsAbvGrd 
                       + TotalBsmtSF*Log1stFlrSF 
                       + LotFrontage*LotAreaSqr 
                       + MSSubClass*Has2Flrs
                       + HouseStyle*Has2Flrs 
                       + LogGrLivArea*LogTotalBath 
                       + Foundation*BsmtCond 
                       + TotalBsmtBath*BsmtTotalFinSFSq 
                       + BsmtCond*BsmtHeight 
                       + GarageCars*OverallQualSqr 
                       + BsmtFinType1*TotalBsmtBath 
                       + YearBuilt*YrSoldAfterRemodel 
                       + OverallQualSqr*LogGrLivArea 
                       + KitchenQual*OverallQualSqr 
                       + BldgType*LotAreaSqr 
                       + GarageArea*OverallQualSqr 
                       + YearBuilt*OverallQualSqr 
                       + YearBuilt*LogTotalBath 
                       + ExterQual*OverallQualSqr 
                       + Foundation*BsmtHeight 
                       + OverallQualSqr*LogTotalBath 
                       + TotalBsmtSF*OverallQualSqr 
                       + TotalBsmtSF*BsmtTotalFinSFSq 
                       + YearBuilt*GarageCars 
                       + OverallQualSqr*YrSoldAfterRemodel 
                       + MSSubClass*HouseStyle 
                       + Has2Flrs*LogGrLivArea 
                       + MasVnrType*MasVnrAreaSqr 
                       + BsmtFinType1*BsmtTotalFinSFSq 
                       + HeatingQC*YrSoldAfterRemodel 
                       + TotalBsmtSF*BsmtHeight 
                       + Log1stFlrSF*LogGrLivArea 
                       + YearBuilt*GarageFinish 
                       + ExterQual*KitchenQual 
                       + GarageCars*LogTotalBath 
                       + LotFrontage*BldgType
                       ##################################
                       + BsmtFinType1*BsmtUnfSF
                       + GarageCars*LogGrLivArea
                       + YearBuilt*GarageArea
                       + HouseStyle*LogGrLivArea
                       + LogTotalBath*LogTotRmsAbvGrd
                       + GarageFinish*OverallQualSqr
                       + YrSoldAfterRemodel*LogTotalBath
                       + Has2Flrs*LogTotRmsAbvGrd
                       + GarageArea*LogTotalBath
                       + LogGrLivArea*FireplacesSqr2
                       + GarageArea*Log1stFlrSF
                       + TotalBsmtSF*GarageArea
                       + GarageArea*LogGrLivArea
                       + BsmtTotalFinSFSq*Log1stFlrSF
                       + YearBuilt*Foundation
                       + BsmtUnfSF*TotalBsmtSF
                       + OverallQualSqr*Log1stFlrSF
                       + TotalBsmtBath*LogTotalBath
                       + BsmtUnfSF*TotalBsmtBath
                       + Foundation*TotalBsmtSF
                       + TotalBsmtSF*GarageCars
                       + HeatingQC*OverallQualSqr
                       + MSZoning*YearBuilt
                       + LotAreaSqr*Log1stFlrSF
                       + MSSubClass*LogGrLivArea
                       + GarageCars*Log1stFlrSF
                       + HouseStyle*LogTotRmsAbvGrd
                       + GarageFinish*LogTotalBath
                       + FireplaceQu*LogGrLivArea
                       + OverallQualSqr*MasVnrAreaSqr
                       + BsmtUnfSF*BsmtTotalFinSFSq
                       + OverallQualSqr*FireplacesSqr2
                       + GarageFinish*YrSoldAfterRemodel
                       + GarageCars*YrSoldAfterRemodel
                       + OverallQualSqr*BsmtHeight
                       + LotFrontage*Log1stFlrSF
                       + OverallQualSqr*LogTotRmsAbvGrd
                       + YearBuilt*OverallCondHigh
                       + YearBuilt*Remodeled
                       + MasVnrType*OverallQualSqr
                       + Neighborhood*OverallQualSqr
                       + BsmtExposure*BsmtTotalFinSFSq
                       + YearBuilt*HeatingQC
                       + YearBuilt*TotalBsmtSF
                       + Log1stFlrSF*LogTotRmsAbvGrd
                       + TotalBsmtSF*LogGrLivArea
                       ,
                       method = "glmboost",
                       trControl = ctrl,
                       tuneGrid = tune,
                       preProc = c("center", "scale"),
                       data = trainExpCont
)

GLMBOOSTpred.trainlog <- predict(GLMBOOSTmodel, newdata = data.frame(trainExpCont))
GLMBOOSTpred.train <- exp(GLMBOOSTpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = GLMBOOSTpred.train)
# 0.1028924 Fitting mstop = 500, prune = yes on full training set

GLMBOOSTpred.testlog <- predict(GLMBOOSTmodel, newdata = data.frame(testExpCont))
GLMBOOSTpred.test <- exp(GLMBOOSTpred.testlog)

Id <- testExpCont_Id
SalePrice <- GLMBOOSTpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2GLMBOOST_Corr1.csv", row.names = FALSE)

#######################################################################
### Use lars: Least Angle Regression, Lasso and Forward Stagewise
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
LARSmodel <- train(LogSalePrice ~. # - StreetPave - HasPool # - NewSale - HasPool - StreetPave - HasGarage
                   + FireplaceQu*FireplacesSqr2
                   + GarageCars*GarageArea 
                   + LogGrLivArea*LogTotRmsAbvGrd 
                   + TotalBsmtSF*Log1stFlrSF 
                   + LotFrontage*LotAreaSqr 
                   + MSSubClass*Has2Flrs
                   + HouseStyle*Has2Flrs 
                   + LogGrLivArea*LogTotalBath 
                   + Foundation*BsmtCond 
                   + TotalBsmtBath*BsmtTotalFinSFSq 
                   + BsmtCond*BsmtHeight 
                   + GarageCars*OverallQualSqr 
                   + BsmtFinType1*TotalBsmtBath 
                   + YearBuilt*YrSoldAfterRemodel 
                   + OverallQualSqr*LogGrLivArea 
                   + KitchenQual*OverallQualSqr 
                   + BldgType*LotAreaSqr 
                   + GarageArea*OverallQualSqr 
                   + YearBuilt*OverallQualSqr 
                   + YearBuilt*LogTotalBath 
                   + ExterQual*OverallQualSqr 
                   + Foundation*BsmtHeight 
                   + OverallQualSqr*LogTotalBath 
                   + TotalBsmtSF*OverallQualSqr 
                   + TotalBsmtSF*BsmtTotalFinSFSq 
                   + YearBuilt*GarageCars 
                   + OverallQualSqr*YrSoldAfterRemodel 
                   + MSSubClass*HouseStyle 
                   + Has2Flrs*LogGrLivArea 
                   + MasVnrType*MasVnrAreaSqr 
                   + BsmtFinType1*BsmtTotalFinSFSq 
                   + HeatingQC*YrSoldAfterRemodel 
                   + TotalBsmtSF*BsmtHeight 
                   + Log1stFlrSF*LogGrLivArea 
                   + YearBuilt*GarageFinish 
                   + ExterQual*KitchenQual 
                   + GarageCars*LogTotalBath 
                   + LotFrontage*BldgType
                   ##################################
                   + BsmtFinType1*BsmtUnfSF
                   + GarageCars*LogGrLivArea
                   + YearBuilt*GarageArea
                   + HouseStyle*LogGrLivArea
                   + LogTotalBath*LogTotRmsAbvGrd
                   + GarageFinish*OverallQualSqr
                   + YrSoldAfterRemodel*LogTotalBath
                   + Has2Flrs*LogTotRmsAbvGrd
                   + GarageArea*LogTotalBath
                   + LogGrLivArea*FireplacesSqr2
                   + GarageArea*Log1stFlrSF
                   + TotalBsmtSF*GarageArea
                   + GarageArea*LogGrLivArea
                   + BsmtTotalFinSFSq*Log1stFlrSF
                   + YearBuilt*Foundation
                   + BsmtUnfSF*TotalBsmtSF
                   + OverallQualSqr*Log1stFlrSF
                   + TotalBsmtBath*LogTotalBath
                   + BsmtUnfSF*TotalBsmtBath
                   + Foundation*TotalBsmtSF
                   + TotalBsmtSF*GarageCars
                   + HeatingQC*OverallQualSqr
                   + MSZoning*YearBuilt
                   + LotAreaSqr*Log1stFlrSF
                   + MSSubClass*LogGrLivArea
                   + GarageCars*Log1stFlrSF
                   + HouseStyle*LogTotRmsAbvGrd
                   + GarageFinish*LogTotalBath
                   + FireplaceQu*LogGrLivArea
                   + OverallQualSqr*MasVnrAreaSqr
                   + BsmtUnfSF*BsmtTotalFinSFSq
                   + OverallQualSqr*FireplacesSqr2
                   + GarageFinish*YrSoldAfterRemodel
                   + GarageCars*YrSoldAfterRemodel
                   + OverallQualSqr*BsmtHeight
                   + LotFrontage*Log1stFlrSF
                   + OverallQualSqr*LogTotRmsAbvGrd
                   + YearBuilt*OverallCondHigh
                   + YearBuilt*Remodeled
                   + MasVnrType*OverallQualSqr
                   + Neighborhood*OverallQualSqr
                   + BsmtExposure*BsmtTotalFinSFSq
                   + YearBuilt*HeatingQC
                   + YearBuilt*TotalBsmtSF
                   + Log1stFlrSF*LogTotRmsAbvGrd
                   + TotalBsmtSF*LogGrLivArea
                   ,
                   method = "lars",
                   trControl = ctrl,
                   preProc = c("center", "scale"),
                   data = trainExpCont
)

LARSpred.trainlog <- predict(LARSmodel, newdata = data.frame(trainExpCont))
LARSpred.train <- exp(LARSpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LARSpred.train)
# 0.09090662 Fitting fraction = 0.525 on full training set
# 0.09252491 Fitting fraction = 0.05 on full training set

LARSpred.testlog <- predict(LARSmodel, newdata = data.frame(testExpCont))
LARSpred.test <- exp(LARSpred.testlog)

Id <- testExpCont_Id
SalePrice <- LARSpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LARS_Corr1.csv", row.names = FALSE)

#######################################################################
### Use SVM: Support Vector Machine, L2 Regularized Linear Kernel
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
SVML2model <- train(LogSalePrice ~.,
                    method = "svmLinear3",
                    trControl = ctrl,
                    preProc = c("center", "scale"),
                    data = trainExpCont
)

SVML2pred.trainlog <- predict(SVML2model, newdata = data.frame(trainExpCont))
SVML2pred.train <- exp(SVML2pred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = SVML2pred.train)

SVML2pred.testlog <- predict(SVML2model, newdata = data.frame(testExpCont))
SVML2pred.test <- exp(SVML2pred.testlog)

Id <- testExpCont_Id
SalePrice <- SVML2pred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2SVML21.csv", row.names = FALSE)

#######################################################################
### Use Logrithmic averaging of LM, GBM, SVML2
#######################################################################
LM_GBM_SVML2_LogAvg.train <- exp((log(LMpred.train) + log(GBMpred.train) + log(SVML2pred.train)) / 3)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LM_GBM_SVML2_LogAvg.train)

LM_GBM_SVML2_LogAvg.test <- exp((log(LMpred.test) + log(GBMpred.test) + log(SVML2pred.test)) / 3)

Id <- testExpCont_Id
SalePrice <- LM_GBM_SVML2_LogAvg.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_GBM_SVML21.csv", row.names = FALSE)

#######################################################################
### Use Logrithmic averaging of LM, GBM, ENET
#######################################################################
LM_GBM_ENET_LogAvg.train <- exp((log(LMpred.train) + log(GBMpred.train) + log(ENETpred.train)) / 3)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LM_GBM_ENET_LogAvg.train)

LM_GBM_ENET_LogAvg.test <- exp((log(LMpred.test) + log(GBMpred.test) + log(ENETpred.test)) / 3)

Id <- testExpCont_Id
SalePrice <- LM_GBM_ENET_LogAvg.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_GBM_ENET1.csv", row.names = FALSE)

#######################################################################
### Use RF
#######################################################################
set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
RFmodel <- train(LogSalePrice ~.,
                 method = "rf",
                 trControl = ctrl,
                 preProc = c("center", "scale"),
                 data = trainExpCont
)

RFpred.trainlog <- predict(RFmodel, newdata = data.frame(trainExpCont))
RFpred.train <- exp(RFpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = RFpred.train)

RFpred.testlog <- predict(RFmodel, newdata = data.frame(testExpCont))
RFpred.test <- exp(RFpred.testlog)

Id <- testExpCont_Id
SalePrice <- RFpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2RF1.csv", row.names = FALSE)

#######################################################################
### Use xgboost using gbtree
#######################################################################
set.seed(1)
require(xgboost)

# For loop to find the best parameters: nfolds?
nrounds_min <- 1
max_depth_min <- 1
eta_min <- 0.001
min_child_weight_min <- 0
rmse_min <- 100 # Some large number
# lambda_min <- 0

for (i in 1:5) { # for max.depth
  for(k in seq(0.001, 0.1, by = 0.001)) { # for eta
    for(j in 1:10) { # for min_child_weight
      # for(l in 1:3) {
      param <- list("objective" = "reg:linear", "eval_metric" = "rmse",
                    "eta" = k, "max.depth" = i, "min_child_weight" = j) # , "lambda" = l)
      bst.cv = xgb.cv(param = param, 
                      data = as.matrix(trainExpContNum), 
                      label = trainExpCont$LogSalePrice, 
                      verbose = 0,
                      lambda = 1,
                      subsample = 0.5,      # Preset
                      nfold = 5,            # Preset
                      nrounds = 500)        # Preset
      
      # Determine minimum rsme error
      min_error <- min(bst.cv$evaluation_log$test_rmse_mean)
      # Determine optimal nrounds_min, max_depth_min, eta_min, rmse_min
      if (min_error < min(rmse_min)) {
        nrounds_min <- c(nrounds_min, which(bst.cv$evaluation_log$test_rmse_mean == min_error))
        max_depth_min <- c(max_depth_min, i)
        eta_min <- c(eta_min, k)
        min_child_weight_min <- c(min_child_weight_min, j)
        # lambda_min <- c(lambda_min, l)
        rmse_min <- c(rmse_min, min_error)
        print(paste("Minimum Error: ", min_error, sep = ""))
        print(paste("Minimum max_depth: ", i, sep = ""))
        print(paste("Minimum eta: ", k, sep = ""))
        print(paste("Minimum min_child_weight: ", j, sep = ""))
        # print(paste("Minimum lambda: ", l, sep = ""))
        print(paste("Minimum nrounds: ", tail(nrounds_min,1), sep = ""))
      }
      # }
    }
  }
}

bst.df <- data.frame(rmse_min = rmse_min, max_depth_min = max_depth_min, 
                     eta_min = eta_min, min_child_weight_min = min_child_weight_min,
                     nrounds_min = nrounds_min)
#################
### Final Test
#################
paramfinal <- list("objective" = "reg:linear", "eval_metric" = "rmse",
                   "eta" = 0.045, "max.depth" = 2, "min_child_weight" = 9)
bstfinal.cv = xgb.cv(param = paramfinal, data = as.matrix(trainExpContNum), 
                     label = trainExpCont$LogSalePrice, 
                     verbose = 0, lambda = 1, subsample = 0.5, nfold = 5,
                     nrounds = 5000)

XGBOOSTModel <- xgboost(data = as.matrix(trainExpContNum), 
                        label = trainExpCont$LogSalePrice, 
                        
                        # min_child_weight = 9, max.depth = 4, eta = 0.05, nround = 277, subsample = 0.5, lambda = 1, 
                        # 0.06835556 housingSimple2xgboost1.csv
                        
                        min_child_weight = 9, max.depth = 2, eta = 0.045, nround = 745, subsample = 0.5, lambda = 1, 
                        # 0.0778001 housingSimple2xgboost2.csv
                        
                        objective = "reg:linear")

XGBOOSTpred.trainlog <- predict(XGBOOSTModel, as.matrix(trainExpContNum))
XGBOOSTpred.train <- exp(XGBOOSTpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = XGBOOSTpred.train)

XGBOOSTpred.testlog <- predict(XGBOOSTModel, as.matrix(testExpContNum))
XGBOOSTpred.test <- exp(XGBOOSTpred.testlog)

Id <- testExpCont_Id
SalePrice <- XGBOOSTpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2xgboost2.csv", row.names = FALSE)

#######################################################################
### Use xgboost using gbLinear
#######################################################################
set.seed(1)
require(xgboost)

# For loop to find the best parameters: nfolds?
nrounds_min <- 1
max_depth_min <- 2
eta_min <- 0.02
min_child_weight_min <- 1
rmse_min <- 100 # Some large number
# lambda_min <- 0

for (i in 2:5) { # for max.depth
  print(paste("Currently working on max_depth: ", i, sep = ""))
  for(k in seq(0.02, 0.1, by = 0.001)) { # for eta
    for(j in 1:10) { # for min_child_weight
      # for(l in 1:3) {
      param <- list("objective" = "reg:linear", "eval_metric" = "rmse", "booster" = "gblinear",
                    "eta" = k, "max.depth" = i, "min_child_weight" = j) # , "lambda" = l)
      bst.cv = xgb.cv(param = param, 
                      data = as.matrix(trainExpContNum), 
                      label = trainExpCont$LogSalePrice, 
                      verbose = 0,
                      lambda = 1,
                      subsample = 0.5,      # Preset
                      nfold = 5,            # Preset
                      nrounds = 500)        # Preset
      
      # Determine minimum rsme error
      min_error <- min(bst.cv$evaluation_log$test_rmse_mean)
      # Determine optimal nrounds_min, max_depth_min, eta_min, rmse_min
      if (min_error < min(rmse_min)) {
        nrounds_min <- c(nrounds_min, which(bst.cv$evaluation_log$test_rmse_mean == min_error))
        max_depth_min <- c(max_depth_min, i)
        eta_min <- c(eta_min, k)
        min_child_weight_min <- c(min_child_weight_min, j)
        # lambda_min <- c(lambda_min, l)
        rmse_min <- c(rmse_min, min_error)
        print(paste("Minimum Error: ", min_error, sep = ""))
        print(paste("Minimum max_depth: ", i, sep = ""))
        print(paste("Minimum eta: ", k, sep = ""))
        print(paste("Minimum min_child_weight: ", j, sep = ""))
        # print(paste("Minimum lambda: ", l, sep = ""))
        print(paste("Minimum nrounds: ", tail(nrounds_min,1), sep = ""))
      }
      # }
    }
  }
}

bst.df <- data.frame(rmse_min = rmse_min, max_depth_min = max_depth_min, 
                     eta_min = eta_min, min_child_weight_min = min_child_weight_min,
                     nrounds_min = nrounds_min)
#################
### Final Test
#################
paramfinal <- list("objective" = "reg:linear", "eval_metric" = "rmse", "booster" = "gblinear",
                   "eta" = 0.045, "max.depth" = 2, "min_child_weight" = 9)
bstfinal.cv = xgb.cv(param = paramfinal, data = as.matrix(trainExpContNum), 
                     label = trainExpCont$LogSalePrice, 
                     verbose = 0, lambda = 1, subsample = 0.5, nfold = 5,
                     nrounds = 5000)

XGBOOST_LINEAR_Model <- xgboost(data = as.matrix(trainExpContNum), 
                                label = trainExpCont$LogSalePrice, 
                                
                                # min_child_weight = 9, max.depth = 4, eta = 0.05, nround = 277, subsample = 0.5, lambda = 1, 
                                #  housingSimple2xgboost_linear1.csv
                                
                                objective = "reg:linear")

XGBOOST_LINEARpred.trainlog <- predict(XGBOOST_LINEAR_Model, as.matrix(trainExpContNum))
XGBOOST_LINEARpred.train <- exp(XGBOOST_LINEARpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = XGBOOST_LINEARpred.train)

XGBOOST_LINEARpred.testlog <- predict(XGBOOST_LINEAR_Model, as.matrix(testExpContNum))
XGBOOST_LINEARpred.test <- exp(XGBOOST_LINEARpred.testlog)

Id <- testExpCont_Id
SalePrice <- XGBOOST_LINEARpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2xgboost_linear1.csv", row.names = FALSE)

##################################################################################
### Use very simplistic stacking technique: Stack LM and GBM and solve using ENET
##################################################################################

trainExpContStack <- data.frame(trainExpCont, LMpred.log = LMpred.trainlog, GBMpred.log = GBMpred.trainlog)
testExpContStack <- data.frame(testExpCont, LMpred.log = LMpred.testlog, GBMpred.log = GBMpred.testlog)

set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
ENET_STACKED_model <- train(LogSalePrice ~.,
                            method = "enet",
                            trControl = ctrl,
                            preProc = c("center", "scale"),
                            data = trainExpContStack
)

ENET_STACKEDpred.trainlog <- predict(ENET_STACKED_model, newdata = data.frame(trainExpContStack))
ENET_STACKEDpred.train <- exp(ENET_STACKEDpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = ENET_STACKEDpred.train)

ENET_STACKEDpred.testlog <- predict(ENET_STACKED_model, newdata = data.frame(testExpContStack))
ENET_STACKEDpred.test <- exp(ENET_STACKEDpred.testlog)

Id <- testExpCont_Id
SalePrice <- ENET_STACKEDpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_GBM_Stacked_ENET1.csv", row.names = FALSE)

##################################################################################
### Use very simplistic stacking technique: Stack LM and GBM and solve using SVML2
##################################################################################

trainExpContStack <- data.frame(trainExpCont, LMpred.log = LMpred.trainlog, GBMpred.log = GBMpred.trainlog)
testExpContStack <- data.frame(testExpCont, LMpred.log = LMpred.testlog, GBMpred.log = GBMpred.testlog)

set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
SVML2_STACKED_model <- train(LogSalePrice ~.,
                             method = "svmLinear3",
                             trControl = ctrl,
                             preProc = c("center", "scale"),
                             data = trainExpContStack
)

SVML2_STACKEDpred.trainlog <- predict(SVML2_STACKED_model, newdata = data.frame(trainExpContStack))
SVML2_STACKEDpred.train <- exp(SVML2_STACKEDpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = SVML2_STACKEDpred.train)

SVML2_STACKEDpred.testlog <- predict(SVML2_STACKED_model, newdata = data.frame(testExpContStack))
SVML2_STACKEDpred.test <- exp(SVML2_STACKEDpred.testlog)

Id <- testExpCont_Id
SalePrice <- SVML2_STACKEDpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_GBM_Stacked_SVML21.csv", row.names = FALSE)

#######################################################################
### Use Logrithmic averaging of LM, GBM, ENET, xgboost Correlation
#######################################################################
LM_GBM_ENET_XGBOOST_LogAvg.train <- exp((log(LMpred.train) + log(GBMpred.train) + log(ENETpred.train) + log(XGBOOSTpred.train)) / 4)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LM_GBM_ENET_XGBOOST_LogAvg.train)
# 0.08227454 housingSimple2LM_GBM_ENET_XGBOOST1.csv
# 0.07710499 housingSimple2LM_GBM_ENET_XGBOOST2.csv

LM_GBM_ENET_XGBOOST_LogAvg.test <- exp((log(LMpred.test) + log(GBMpred.test) + log(ENETpred.test) + log(XGBOOSTpred.test)) / 4)

Id <- testExpCont_Id
SalePrice <- LM_GBM_ENET_XGBOOST_LogAvg.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_GBM_ENET_XGBOOST2.csv", row.names = FALSE)

#######################################################################
### Use Logrithmic averaging of LM, GBM, bayesglm, xgboost Correlation
#######################################################################
LM_GBM_BAYESGLM_XGBOOST_LogAvg.train <- exp((log(LMpred.train) + log(GBMpred.train) + log(BAYESGLMpred.train) + log(XGBOOSTpred.train)) / 4)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LM_GBM_BAYESGLM_XGBOOST_LogAvg.train)
# 0.08193796 housingSimple2LM_GBM_BAYESGLM_XGBOOST1.csv
# 0.07629284 housingSimple2LM_GBM_BAYESGLM_XGBOOST2.csv

LM_GBM_BAYESGLM_XGBOOST_LogAvg.test <- exp((log(LMpred.test) + log(GBMpred.test) + log(BAYESGLMpred.test) + log(XGBOOSTpred.test)) / 4)

Id <- testExpCont_Id
SalePrice <- LM_GBM_BAYESGLM_XGBOOST_LogAvg.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_GBM_BAYESGLM_XGBOOST2.csv", row.names = FALSE)

############################################################################################
### Use Logrithmic averaging of LM, GBM, ENET, xgboost, ridge, bayesglm, lars, glmnet
############################################################################################
LM_GBM_ENET_XGBOOST_RIDGE_BAYESGLM_LARS_GLMNET_LogAvg.train <- exp((log(LMpred.train) + 
                                                                      log(GBMpred.train) + 
                                                                      log(ENETpred.train) + 
                                                                      log(XGBOOSTpred.train) +
                                                                      log(RIDGEpred.train) +
                                                                      log(BAYESGLMpred.train) +
                                                                      log(LARSpred.train) +
                                                                      log(GLMNETpred.train)
) / 8)
myRMSLE(actual = trainExpCont_SalePrice, predicted = LM_GBM_ENET_XGBOOST_RIDGE_BAYESGLM_LARS_GLMNET_LogAvg.train)
# 0.08597919 housingSimple2LM_GBM_ENET_XGBOOST_RIDGE_BAYESGLM_LARS_GLMNET_Corr1.csv
# 0.08306808 housingSimple2LM_GBM_ENET_XGBOOST_RIDGE_BAYESGLM_LARS_GLMNET_Corr2.csv

LM_GBM_ENET_XGBOOST_RIDGE_BAYESGLM_LARS_GLMNET_LogAvg.test <- exp((log(LMpred.test) + 
                                                                     log(GBMpred.test) + 
                                                                     log(ENETpred.test) + 
                                                                     log(XGBOOSTpred.test) +
                                                                     log(RIDGEpred.test) +
                                                                     log(BAYESGLMpred.test) +
                                                                     log(LARSpred.test) +
                                                                     log(GLMNETpred.test)
) / 8)
Id <- testExpCont_Id
SalePrice <- LM_GBM_ENET_XGBOOST_RIDGE_BAYESGLM_LARS_GLMNET_LogAvg.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_GBM_ENET_XGBOOST_RIDGE_BAYESGLM_LARS_GLMNET_Corr2.csv", row.names = FALSE)

##################################################################################
### Use stacking technique: Stack LM and XGBOOST, solve using GBM
##################################################################################
LM_XGBOOSTpred.trainlog <- (LMpred.trainlog + XGBOOSTpred.trainlog) / 2
LM_XGBOOSTpred.testlog <- (LMpred.testlog + XGBOOSTpred.testlog) / 2

trainExpContStack <- data.frame(trainExpCont, StackedPred.log = LM_XGBOOSTpred.trainlog) # LMpred.log = LMpred.trainlog, GBMpred.log = GBMpred.trainlog)
testExpContStack <- data.frame(testExpCont, StackedPred.log = LM_XGBOOSTpred.testlog) # LMpred.log = LMpred.testlog, GBMpred.log = GBMpred.testlog)

set.seed(1)
ctrl <- trainControl(method = "repeatedcv", repeats = 10, number = 5, verboseIter = TRUE)
tune <- expand.grid(n.trees = seq(150, 450, by = 100), interaction.depth = c(1, 2, 3),
                    n.minobsinnode = 10, shrinkage = c(0.1, 0.2))
GBM_STACKED_model <- train(LogSalePrice ~. # - NewSale - HasPool - StreetPave - HasGarage
                           + FireplaceQu*FireplacesSqr2
                           + GarageCars*GarageArea 
                           + LogGrLivArea*LogTotRmsAbvGrd 
                           + TotalBsmtSF*Log1stFlrSF 
                           + LotFrontage*LotAreaSqr 
                           + MSSubClass*Has2Flrs
                           + HouseStyle*Has2Flrs 
                           + LogGrLivArea*LogTotalBath 
                           + Foundation*BsmtCond 
                           + TotalBsmtBath*BsmtTotalFinSFSq 
                           + BsmtCond*BsmtHeight 
                           + GarageCars*OverallQualSqr 
                           + BsmtFinType1*TotalBsmtBath 
                           + YearBuilt*YrSoldAfterRemodel 
                           + OverallQualSqr*LogGrLivArea 
                           + KitchenQual*OverallQualSqr 
                           + BldgType*LotAreaSqr 
                           + GarageArea*OverallQualSqr 
                           + YearBuilt*OverallQualSqr 
                           + YearBuilt*LogTotalBath 
                           + ExterQual*OverallQualSqr 
                           + Foundation*BsmtHeight 
                           + OverallQualSqr*LogTotalBath 
                           + TotalBsmtSF*OverallQualSqr 
                           + TotalBsmtSF*BsmtTotalFinSFSq 
                           + YearBuilt*GarageCars 
                           + OverallQualSqr*YrSoldAfterRemodel 
                           + MSSubClass*HouseStyle 
                           + Has2Flrs*LogGrLivArea 
                           + MasVnrType*MasVnrAreaSqr 
                           + BsmtFinType1*BsmtTotalFinSFSq 
                           + HeatingQC*YrSoldAfterRemodel 
                           + TotalBsmtSF*BsmtHeight 
                           + Log1stFlrSF*LogGrLivArea 
                           + YearBuilt*GarageFinish 
                           + ExterQual*KitchenQual 
                           + GarageCars*LogTotalBath 
                           + LotFrontage*BldgType
                           ##################################
                           + BsmtFinType1*BsmtUnfSF
                           + GarageCars*LogGrLivArea
                           + YearBuilt*GarageArea
                           + HouseStyle*LogGrLivArea
                           + LogTotalBath*LogTotRmsAbvGrd
                           + GarageFinish*OverallQualSqr
                           + YrSoldAfterRemodel*LogTotalBath
                           + Has2Flrs*LogTotRmsAbvGrd
                           + GarageArea*LogTotalBath
                           + LogGrLivArea*FireplacesSqr2
                           + GarageArea*Log1stFlrSF
                           + TotalBsmtSF*GarageArea
                           + GarageArea*LogGrLivArea
                           + BsmtTotalFinSFSq*Log1stFlrSF
                           + YearBuilt*Foundation
                           + BsmtUnfSF*TotalBsmtSF
                           + OverallQualSqr*Log1stFlrSF
                           + TotalBsmtBath*LogTotalBath
                           + BsmtUnfSF*TotalBsmtBath
                           + Foundation*TotalBsmtSF
                           + TotalBsmtSF*GarageCars
                           + HeatingQC*OverallQualSqr
                           + MSZoning*YearBuilt
                           + LotAreaSqr*Log1stFlrSF
                           + MSSubClass*LogGrLivArea
                           + GarageCars*Log1stFlrSF
                           + HouseStyle*LogTotRmsAbvGrd
                           + GarageFinish*LogTotalBath
                           + FireplaceQu*LogGrLivArea
                           + OverallQualSqr*MasVnrAreaSqr
                           + BsmtUnfSF*BsmtTotalFinSFSq
                           + OverallQualSqr*FireplacesSqr2
                           + GarageFinish*YrSoldAfterRemodel
                           + GarageCars*YrSoldAfterRemodel
                           + OverallQualSqr*BsmtHeight
                           + LotFrontage*Log1stFlrSF
                           + OverallQualSqr*LogTotRmsAbvGrd
                           + YearBuilt*OverallCondHigh
                           + YearBuilt*Remodeled
                           + MasVnrType*OverallQualSqr
                           + Neighborhood*OverallQualSqr
                           + BsmtExposure*BsmtTotalFinSFSq
                           + YearBuilt*HeatingQC
                           + YearBuilt*TotalBsmtSF
                           + Log1stFlrSF*LogTotRmsAbvGrd
                           + TotalBsmtSF*LogGrLivArea
                           ,
                           method = "gbm",
                           trControl = ctrl,
                           tuneGrid = tune,
                           preProc = c("center", "scale"),
                           data = trainExpContStack
)

GBM_STACKEDpred.trainlog <- predict(GBM_STACKED_model, newdata = data.frame(trainExpContStack))
GBM_STACKEDpred.train <- exp(GBM_STACKEDpred.trainlog)
myRMSLE(actual = trainExpCont_SalePrice, predicted = GBM_STACKEDpred.train)
# 0.08124273

GBM_STACKEDpred.testlog <- predict(GBM_STACKED_model, newdata = data.frame(testExpContStack))
GBM_STACKEDpred.test <- exp(GBM_STACKEDpred.testlog)

Id <- testExpCont_Id
SalePrice <- GBM_STACKEDpred.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimple2LM_XGBOOST_Stacked_GBM_Corr1.csv", row.names = FALSE)

######################
# Using PCA regression
######################

# LM_GBM_PCALogAvg.train <- exp((log(LM_PCApred.train) + log(GBM_PCApred.train)) / 2)
# myRMSLE(actual = trainExpCont_SalePrice, predicted = LM_GBM_PCALogAvg.train)
# 
# LM_GBM_PCALogAvg.test <- exp((log(LM_PCApred.test) + log(GBM_PCApred.test)) / 2)
# 
# Id <- testExpCont_Id
# SalePrice <- LM_GBM_PCALogAvg.test
# output <- data.frame(Id, SalePrice)
# write.csv(output, file = "housingSimpleLM_GBM_PCA.csv", row.names = FALSE)

#######################################################################
### Write out to file
#######################################################################
#Id <- testFinalRF$Id
#SalePrice <- RFTestpred2.test
#output <- data.frame(Id, SalePrice)
#write.csv(output, file = "housingSimple.csv", row.names = FALSE)
