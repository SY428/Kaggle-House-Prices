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

trainExp$Residential <- ifelse(trainExp$MSZoning != "C (all)", 1, 0)
trainExp$Residential <- as.factor(trainExp$Residential)
trainExp$MSZoning <- NULL

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

trainExp$AlleyNA <- ifelse(is.na(trainExp$Alley), 1, 0)
trainExp$AlleyNA <- as.factor(trainExp$AlleyNA)
trainExp$AlleyPave <- ifelse((trainExp$Alley == "Pave") & !is.na(trainExp$Alley), 1, 0)
trainExp$AlleyPave <- as.factor(trainExp$AlleyPave)
trainExp$Alley <- NULL

trainExp$LotShapeReg <- ifelse(trainExp$LotShape == "Reg", 1, 0)
trainExp$LotShapeReg <- as.factor(trainExp$LotShapeReg)
trainExp$LotShape <- NULL

trainExp$LandContourHLS <- ifelse(trainExp$LandContour == "HLS", 1, 0)
trainExp$LandContourLow <- ifelse(trainExp$LandContour == "Low", 1, 0)
trainExp$LandContourLvl <- ifelse(trainExp$LandContour == "Lvl", 1, 0)
trainExp$LandContour <- NULL

trainExp$Utilities <- NULL

trainExp$LandSlopeMod <- ifelse(trainExp$LandSlope == "Mod", 1, 0)
trainExp$LandSlopeSev <- ifelse(trainExp$LandSlope == "Sev", 1, 0)
trainExp$LandSlope <- NULL

trainExp$LotConfig <- as.factor(trainExp$LotConfig) # Ordered Factor

trainExp$Neighborhood <- as.factor(trainExp$Neighborhood) # Ordered Factor

trainExp$Condition1 <- as.factor(trainExp$Condition1) # Ordered Factor

trainExp$Condition2 <- NULL

trainExp$BldgType1Fam <- ifelse(trainExp$BldgType == "1Fam", 1, 0)
trainExp$BldgType <- NULL

trainExp$HouseStyleHiPrice <- ifelse((trainExp$HouseStyle == "2.5Fin") | (trainExp$HouseStyle == "2Story"), 1, 0)
trainExp$HouseStyleAbAvPrice <- ifelse((trainExp$HouseStyle == "1Story") | (trainExp$HouseStyle == "SLvl"), 1, 0)
trainExp$HouseStyleAvPrice <- ifelse((trainExp$HouseStyle == "1.5Fin") | (trainExp$HouseStyle == "2.5Unf") | (trainExp$HouseStyle == "SFoyer"), 1, 0)
# trainExp$HouseStyleLowPrice <- ifelse(trainExp$HouseStyle == "1.5Unf", 1, 0)
trainExp$HouseStyle <- NULL

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
trainExp$OverallCondLow <- ifelse(trainExp$OverallCond < 4.5, 1, 0)
trainExp$OverallCond <- NULL

trainExp$Remodeled <- ifelse(trainExp$YearBuilt == trainExp$YearRemodAdd, 1, 0)
trainExp$YrSoldAfterRemodel <- trainExp$YrSold - trainExp$YearRemodAdd
trainExp$YrSoldAfterRemodel <- as.numeric(trainExp$YrSoldAfterRemodel)
trainExp$YearRemodAdd <- NULL

trainExp$YearBuilt <- as.numeric(trainExp$YearBuilt)

trainExp$RoofStyle <- NULL

trainExp$RoofMatl <- NULL

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
from <- c("NA", "Fa", "TA", "Gd", "Ex")
to <- c(11.53, 11.63, 11.81, 12.16, 12.67) - 11.53
levels(trainExp$BsmtHeight) <- c("Ex", "Fa", "Gd", "TA", "NA")
trainExp$BsmtHeight[which(is.na(trainExp$BsmtHeight))] <- "NA"
trainExp$BsmtHeight <- as.numeric(as.character(mapvalues(trainExp$BsmtHeight, to=to, from=from)))

trainExp$TotalBsmtBath <- trainExp$BsmtFullBath + (0.5 * trainExp$BsmtHalfBath)
#trainExp$BsmtFullBath <- NULL
#trainExp$BsmtHalfBath <- NULL

trainExp$BsmtCond <- NULL

trainExp$BsmtExposure <- NULL

trainExp$BsmtFinType2 <- NULL

trainExp$BsmtFinType1 <- as.factor(trainExp$BsmtFinType1) # Ordered factor

trainExp$BsmtTotalFinSF <- trainExp$BsmtFinSF1 + trainExp$BsmtFinSF2
trainExp$BsmtTotalFinSFSq <- trainExp$BsmtTotalFinSF^2
trainExp$BsmtTotalFinSF <- NULL
trainExp$BsmtFinSF1 <- NULL
trainExp$BsmtFinSF2 <- NULL

# has NA
# trainExp$BsmtUnfSF

trainExp$Heating <- NULL

trainExp$HeatingQC <- as.factor(trainExp$HeatingQC) # Ordered Factors

trainExp$CentralAirYes <- ifelse(trainExp$CentralAir == "Y", 1, 0)
trainExp$CentralAir <- NULL

trainExp$ElectricalSBrkr <- ifelse((trainExp$Electrical == "SBrkr") | is.na(trainExp$Electrical), 1, 0)
trainExp$Electrical <- NULL

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

trainExp$GarageType <- NULL

trainExp$HasGarage <- ifelse(!is.na(trainExp$GarageYrBlt), 1, 0)
# trainExp$GarageRebuilt <- ifelse(trainExp$GarageYrBlt != trainExp$YearBuilt, 1, 0)
trainExp$GarageYrBlt <- NULL

trainExp$GarageFinish <- NULL

trainExp$GarageCars <- as.numeric(trainExp$GarageCars)

plot(trainExp$LogSalePrice~(trainExp$GarageArea))
logregModel <- lm(trainExp$LogSalePrice~trainExp$GarageArea)
abline(logregModel)
trainExp$GarageArea <- as.numeric(trainExp$GarageArea)

trainExp$GarageCond <- NULL

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
trainExp$PoolArea <- NULL
trainExp$PoolQC <- NULL

trainExp$HasFence <- ifelse(!is.na(trainExp$Fence), 1, 0)
trainExp$Fence <- NULL

trainExp$MiscFeature <- NULL

trainExp$MiscVal <- NULL

trainExp$MoSold <- NULL

trainExp$YrSold <- NULL

trainExp$NewSale <- ifelse(trainExp$SaleType == "New", 1, 0)
trainExp$ConSale <- ifelse((trainExp$SaleType == "Con") | (trainExp$SaleType == "ConLD") | (trainExp$SaleType == "ConLI"), 1, 0)
trainExp$SaleType <- NULL

trainExp$SaleCondPartial <- ifelse(trainExp$SaleCondition == "Partial", 1, 0)
trainExp$SaleCondNormal <- ifelse(trainExp$SaleCondition == "Normal", 1, 0)
trainExp$SaleCondition <- NULL

### Deal with ordered factors
ordered.factors <- c("MSSubClass", "LotConfig", "Neighborhood", "Condition1", 
                     "ExterQual", "ExterCond", "Foundation", "HeatingQC", "PavedDrive",
                     "MasVnrType",
                     
                     "Exterior1st", "Exterior2nd", "BsmtFinType1", "KitchenQual",
                     "Functional", "FireplaceQu", "GarageQual")

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

testExp$Residential <- ifelse(testExp$MSZoning != "C (all)" | is.na(testExp$MSZoning), 1, 0)
# testExp$Residential <- as.factor(testExp$Residential)
testExp$MSZoning <- NULL

testExp$LotFrontage <- as.numeric(testExp$LotFrontage)
testExp$LotArea <- as.numeric(testExp$LotArea)

predFrontage <- LotModel$coefficients[1] + testExp$LotArea[which(is.na(testExp$LotFrontage))]*LotModel$coefficients[2]

testExp$LotFrontage[which(is.na(testExp$LotFrontage))] <- predFrontage
plot(y = testExp$LotFrontage, x = testExp$LotArea)

testExp$LotAreaSqr <- sqrt(testExp$LotArea)
testExp$LotArea <- NULL

testExp$StreetPave <- ifelse(testExp$Street == "Pave", 1, 0)
testExp$Street <- NULL

testExp$AlleyNA <- ifelse(is.na(testExp$Alley), 1, 0)
testExp$AlleyPave <- ifelse((testExp$Alley == "Pave") & !is.na(testExp$Alley), 1, 0)
testExp$Alley <- NULL

testExp$LotShapeReg <- ifelse(testExp$LotShape == "Reg", 1, 0)
testExp$LotShape <- NULL

testExp$LandContourHLS <- ifelse(testExp$LandContour == "HLS", 1, 0)
testExp$LandContourLow <- ifelse(testExp$LandContour == "Low", 1, 0)
testExp$LandContourLvl <- ifelse(testExp$LandContour == "Lvl", 1, 0)
testExp$LandContour <- NULL

testExp$Utilities <- NULL

testExp$LandSlopeMod <- ifelse(testExp$LandSlope == "Mod", 1, 0)
testExp$LandSlopeSev <- ifelse(testExp$LandSlope == "Sev", 1, 0)
testExp$LandSlope <- NULL

testExp$LotConfig <- as.factor(testExp$LotConfig) # Ordered Factor

testExp$Neighborhood <- as.factor(testExp$Neighborhood) # Ordered Factor

testExp$Condition1 <- as.factor(testExp$Condition1) # Ordered Factor

testExp$Condition2 <- NULL

testExp$BldgType1Fam <- ifelse(testExp$BldgType == "1Fam", 1, 0)
testExp$BldgType <- NULL

testExp$HouseStyleHiPrice <- ifelse((testExp$HouseStyle == "2.5Fin") | (testExp$HouseStyle == "2Story"), 1, 0)
testExp$HouseStyleAbAvPrice <- ifelse((testExp$HouseStyle == "1Story") | (testExp$HouseStyle == "SLvl"), 1, 0)
testExp$HouseStyleAvPrice <- ifelse((testExp$HouseStyle == "1.5Fin") | (testExp$HouseStyle == "2.5Unf") | (testExp$HouseStyle == "SFoyer"), 1, 0)
# testExp$HouseStyleLowPrice <- ifelse(testExp$HouseStyle == "1.5Unf", 1, 0)
testExp$HouseStyle <- NULL

testExp$OverallQualSqr <- testExp$OverallQual^2
testExp$OverallQualSqr <- as.numeric(testExp$OverallQualSqr)
testExp$OverallQual <- NULL

testExp$OverallCondHigh <- ifelse(testExp$OverallCond > 6.5, 1, 0)
testExp$OverallCondLow <- ifelse(testExp$OverallCond < 4.5, 1, 0)
testExp$OverallCond <- NULL

testExp$YearBuilt <- as.numeric(testExp$YearBuilt)

testExp$Remodeled <- ifelse(testExp$YearBuilt == testExp$YearRemodAdd, 1, 0)
testExp$YrSoldAfterRemodel <- testExp$YrSold - testExp$YearRemodAdd
testExp$YrSoldAfterRemodel <- as.numeric(testExp$YrSoldAfterRemodel)
testExp$YearRemodAdd <- NULL

testExp$RoofStyle <- NULL

testExp$RoofMatl <- NULL

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
from <- c("NA", "Fa", "TA", "Gd", "Ex")
to <- c(11.53, 11.63, 11.81, 12.16, 12.67) - 11.53
levels(testExp$BsmtHeight) <- c("Ex", "Fa", "Gd", "TA", "NA")
testExp$BsmtHeight[which(is.na(testExp$BsmtHeight))] <- "NA"
testExp$BsmtHeight <- as.numeric(as.character(mapvalues(testExp$BsmtHeight, to=to, from=from)))

# NAs in bathrooms set to 0
testExp$BsmtFullBath[which(is.na(testExp$BsmtFullBath))] <- 0
testExp$BsmtHalfBath[which(is.na(testExp$BsmtHalfBath))] <- 0
testExp$TotalBsmtBath <- testExp$BsmtFullBath + (0.5 * testExp$BsmtHalfBath)

testExp$BsmtCond <- NULL

testExp$BsmtExposure <- NULL

testExp$BsmtFinType2 <- NULL

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

testExp$Heating <- NULL

testExp$HeatingQC <- as.factor(testExp$HeatingQC) # Ordered Factors

testExp$CentralAirYes <- ifelse(testExp$CentralAir == "Y", 1, 0)
testExp$CentralAir <- NULL

testExp$ElectricalSBrkr <- ifelse((testExp$Electrical == "SBrkr") | is.na(testExp$Electrical), 1, 0)
testExp$Electrical <- NULL

testExp$Log1stFlrSF <- log(testExp$X1stFlrSF+1)
testExp$X1stFlrSF <- NULL

testExp$Has2Flrs <- ifelse(testExp$X2ndFlrSF > 0, 1, 0)
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

testExp$GarageType <- NULL

which(testExp$GarageYrBlt>2010)
testExp$GarageYrBlt[1133] <- 2007
testExp$HasGarage <- ifelse(!is.na(testExp$GarageYrBlt), 1, 0)
# testExp$GarageRebuilt <- ifelse(testExp$GarageYrBlt != testExp$YearBuilt, 1, 0)
testExp$GarageYrBlt <- NULL

testExp$GarageFinish <- NULL

testExp$GarageCars[which(is.na(testExp$GarageCars))] <- 0
testExp$GarageCars <- as.numeric(testExp$GarageCars)

testExp$GarageArea[which(is.na(testExp$GarageArea))] <- 0
testExp$GarageArea <- as.numeric(testExp$GarageArea)

testExp$GarageCond <- NULL

testExp$GarageQual <- as.factor(testExp$GarageQual) # Ordered Factor

testExp$PavedDrive <- as.factor(testExp$PavedDrive) # Ordered Factor

testExp$TotalPorchDeckArea <- testExp$WoodDeckSF + testExp$OpenPorchSF + testExp$EnclosedPorch + testExp$X3SsnPorch + testExp$ScreenPorch
testExp$WoodDeckSF <- NULL
testExp$OpenPorchSF <- NULL
testExp$EnclosedPorch <- NULL
testExp$X3SsnPorch <- NULL
testExp$ScreenPorch <- NULL

testExp$HasPool <- ifelse(testExp$PoolArea > 0, 1, 0)
testExp$PoolArea <- NULL
testExp$PoolQC <- NULL

testExp$HasFence <- ifelse(!is.na(testExp$Fence), 1, 0)
testExp$Fence <- NULL

testExp$MiscFeature <- NULL

testExp$MiscVal <- NULL

testExp$MoSold <- NULL

testExp$YrSold <- NULL

testExp$NewSale <- ifelse(testExp$SaleType == "New", 1, 0)
testExp$ConSale <- ifelse((testExp$SaleType == "Con") | (testExp$SaleType == "ConLD") | (testExp$SaleType == "ConLI"), 1, 0)
testExp$SaleType <- NULL
testExp$NewSale[1030] <- 0
testExp$ConSale[1030] <- 0

testExp$SaleCondPartial <- ifelse(testExp$SaleCondition == "Partial", 1, 0)
testExp$SaleCondNormal <- ifelse(testExp$SaleCondition == "Normal", 1, 0)
testExp$SaleCondition <- NULL

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
LMtrainSelModel0 <- lm(LogSalePrice ~ YearBuilt + Residential + LotAreaSqr + OverallQualSqr 
                                    + OverallCondHigh + OverallCondLow + YrSoldAfterRemodel
                                    + LogGrLivArea + TotalPorchDeckArea + SaleCondNormal, 
                       trainExpRemove)
LMtrainSelModel1 <- lm(LogSalePrice ~ YearBuilt + LotAreaSqr + OverallQualSqr 
                       + OverallCondHigh + OverallCondLow
                       + LogGrLivArea + TotalPorchDeckArea, 
                       trainExpRemove)

## Deal with ordered factors
ordered.factors <- c("MSSubClass", "LotConfig", "Neighborhood", "Condition1",
                     "ExterQual", "ExterCond", "Foundation", "HeatingQC", "PavedDrive",
                     "Exterior1st", "Exterior2nd", "BsmtFinType1", "KitchenQual",
                     "Functional", "FireplaceQu", "GarageQual", "MasVnrType")

BaseModel <- "LogSalePrice ~ YearBuilt + LotAreaSqr + OverallQualSqr + OverallCondHigh + OverallCondLow + LogGrLivArea + TotalPorchDeckArea"

MakeFactorContinuous <- function(df, factors, bm) {
  factor_list <- list()
  basemodel <- bm
  for (i in 1:length(factors)) {
    formula <- paste(basemodel, " + ", factors[i], " -1", sep = "")
    new_model <- lm(formula = formula, data = df)
    coef_ind <- grep(factors[i], names(new_model$coefficients))
  
    from <- gsub(factors[i], "", names(new_model$coefficients[coef_ind]))
    to <- as.numeric(new_model$coefficients[coef_ind])
  
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
trainExpCont$Residential <- as.numeric(trainExpCont$Residential)

testExpCont$SalePrice <- NULL
testExpCont$Id <- NULL
testExpCont$Residential <- as.numeric(testExpCont$Residential)

LMtrainContModel2 <- lm(LogSalePrice~., trainExpCont)

############################
## Tweak Existing Predictors
############################

feature_removed <- c()
# feature_removed <- c("LotFrontage", "FireplaceQu", "LandContourLvl", 
#                      "HouseStyleHiPrice", "HouseStyleAvPrice", "Has2Flrs", 
#                      "SaleCondPartial")
trainExpCont <- trainExpCont[!(names(trainExpCont) %in% feature_removed)]
testExpCont <- testExpCont[!(names(testExpCont) %in% feature_removed)]

LMtrainContModel3 <- lm(LogSalePrice~., trainExpCont)

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
LMmodel <- train(LogSalePrice ~ .,
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
write.csv(output, file = "housingSimpleLM3.csv", row.names = FALSE)

######################
# Using PCA regression
######################
# set.seed(1)
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# LM_PCAmodel <- train(LogSalePrice ~ .,
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
GBMmodel <- train(LogSalePrice ~ .,
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
write.csv(output, file = "housingSimpleGBM3.csv", row.names = FALSE)

######################
# Using PCA regression
######################
# set.seed(1)
# ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 5, verboseIter = TRUE)
# GBM_PCAmodel <- train(LogSalePrice ~ .,
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

LM_GBM_LogAvg.test <- exp((log(LMpred.test) + log(GBMpred.test)) / 2)

Id <- testExpCont_Id
SalePrice <- LM_GBM_LogAvg.test
output <- data.frame(Id, SalePrice)
write.csv(output, file = "housingSimpleLM_GBM3.csv", row.names = FALSE)

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
