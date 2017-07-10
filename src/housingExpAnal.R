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

#####################################################################
### Analyzing the predictors
### Exploratory Analysis
#####################################################################
par(mfrow=c(1,1))
histogram(train$MSSubClass); histogram(test$MSSubClass);
plot(x=train$YearBuilt, y=as.numeric(train$MSSubClass))
plot(x=test$YearBuilt, y=as.numeric(test$MSSubClass))
plot(y=train$SalePrice, x=train$MSSubClass)
### Nothing too observable ###

plot(x=train$YearBuilt, y=train$LotArea)
par(mfrow=c(1,2))
boxplot(train$LotArea, ylim=c(0,max(train$LotArea)))
boxplot(test$LotArea, ylim=c(0,max(train$LotArea)))
plot(y=train$SalePrice, x=train$LotArea)
### Remove training points with LotArea > 56600 (60000) ###
### Relatinship with price

par(mfrow=c(2,1))
plot(x=train$YearBuilt, y=train$Street)
plot(x=test$YearBuilt, y=test$Street)
plot(y=train$SalePrice, x=train$Street)
summary(subset(train, train$Street == "Grvl")$SalePrice)
summary(subset(train, train$Street == "Pave")$SalePrice)
### Only 6 Grvl Streets on test/training data ###

par(mfrow=c(1,2))
histogram(train$LotShape); histogram(test$LotShape);
plot(x=train$YearBuilt, y=train$LotShape)
plot(x=test$YearBuilt, y=test$LotShape)
plot(y=train$SalePrice, x=train$LotShape)
### Mostly Reg and IR1 ###

par(mfrow=c(1,2))
histogram(train$LandContour); histogram(test$LandContour);
plot(x=train$YearBuilt, y=train$LandContour)
plot(x=test$YearBuilt, y=test$LandContour)
plot(y=train$SalePrice, x=train$LandContour)
### Mostly Lvl ###

par(mfrow=c(1,2))
histogram(train$LotConfig); histogram(test$LotConfig);
plot(x=train$YearBuilt, y=train$LotConfig)
plot(x=test$YearBuilt, y=test$LotConfig)
plot(y=train$SalePrice, x=train$LotConfig)
### Mostly Inside ###

par(mfrow=c(1,2))
histogram(train$LandSlope); histogram(test$LandSlope);
plot(x=train$YearBuilt, y=train$LandSlope)
plot(x=test$YearBuilt, y=test$LandSlope)
plot(y=train$SalePrice, x=train$LandSlope)
### Mostly Gtl ###

par(mfrow=c(1,2))
histogram(train$Neighborhood); histogram(test$Neighborhood);
plot(x=train$YearBuilt, y=train$Neighborhood)
plot(x=test$YearBuilt, y=test$Neighborhood)
plot(y=train$SalePrice, x=train$Neighborhood)
### Maybe split by Neighborhood ? ###

par(mfrow=c(1,2))
histogram(train$Condition1); histogram(test$Condition1);
plot(x=train$YearBuilt, y=train$Condition1)
plot(x=test$YearBuilt, y=test$Condition1)
plot(y=train$SalePrice, x=train$Condition1)
### Mostly norm
### No real correlation ###

par(mfrow=c(1,2))
histogram(train$Condition2); histogram(test$Condition2);
plot(x=train$YearBuilt, y=train$Condition2)
plot(x=test$YearBuilt, y=test$Condition2)
plot(y=train$SalePrice, x=train$Condition2)
### Mostly Norm, 98%? ###

par(mfrow=c(1,1))
histogram(train$BldgType); histogram(test$BldgType);
plot(x=train$YearBuilt, y=train$BldgType)
plot(x=test$YearBuilt, y=test$BldgType)
plot(y=train$SalePrice, x=train$BldgType)
### Mostly 1Fam ###

par(mfrow=c(1,2))
histogram(train$HouseStyle); histogram(test$HouseStyle);
plot(x=train$YearBuilt, y=train$HouseStyle)
plot(x=test$YearBuilt, y=test$HouseStyle)
plot(y=train$SalePrice, x=train$HouseStyle)
### No relationship here ###

par(mfrow=c(1,2))
histogram(train$OverallQual); histogram(test$OverallQual);
plot(x=train$YearBuilt, y=train$OverallQual)
plot(x=test$YearBuilt, y=test$OverallQual)
plot(y=train$SalePrice, x=train$OverallQual)
### Nothing here ###
### trend with price

par(mfrow=c(1,2))
histogram(train$OverallCond); histogram(test$OverallCond);
plot(x=train$YearBuilt, y=train$OverallCond)
plot(x=test$YearBuilt, y=test$OverallCond)
plot(y=train$SalePrice, x=train$OverallCond)
### Nothing here ###

par(mfrow=c(1,1))
histogram(train$YearBuilt); histogram(test$YearBuilt);
plot(y=train$SalePrice, x=train$YearBuilt)
### Nothing here ###
### Trend with price
### Remove a few outliers

histogram(train$YearRemodAdd); histogram(test$YearRemodAdd);
plot(x=train$YearBuilt, y=train$YearRemodAdd)
plot(x=test$YearBuilt, y=test$YearRemodAdd)
plot(y=train$SalePrice, x=train$YearRemodAdd)
### Year built vs remodeled? ###

histogram(train$RoofStyle); histogram(test$RoofStyle);
plot(x=train$YearBuilt, y=train$RoofStyle)
plot(x=test$YearBuilt, y=test$RoofStyle)
plot(y=train$SalePrice, x=train$RoofStyle)
### Mostly 2 styles ###

histogram(train$RoofMatl)
histogram(test$RoofMatl)
plot(x=train$YearBuilt, y=train$RoofMatl)
plot(x=test$YearBuilt, y=test$RoofMatl)
plot(y=train$SalePrice, x=train$RoofMatl)
### Mostly CompShg

histogram(train$ExterQual)
histogram(test$ExterQual)
plot(x=train$YearBuilt, y=train$ExterQual)
plot(x=test$YearBuilt, y=test$ExterQual)
plot(y=train$SalePrice, x=train$ExterQual)
### Generally 2
### Sort of a trend

histogram(train$ExterCond)
histogram(test$ExterCond)
plot(x=train$YearBuilt, y=train$ExterCond)
plot(x=test$YearBuilt, y=test$ExterCond)
plot(y=train$SalePrice, x=train$ExterCond)
### Generally 2

histogram(train$Foundation)
histogram(test$Foundation)
plot(x=train$YearBuilt, y=train$Foundation)
plot(x=test$YearBuilt, y=test$Foundation)
plot(y=train$SalePrice, x=train$Foundation)
### Nothing

histogram(train$Heating)
histogram(test$Heating)
plot(x=train$YearBuilt, y=train$Heating)
plot(x=test$YearBuilt, y=test$Heating)
plot(y=train$SalePrice, x=train$Heating)
### Mostly GasA

par(mfrow=c(1,2))
histogram(train$HeatingQC)
histogram(test$HeatingQC)
plot(x=train$YearBuilt, y=train$HeatingQC)
plot(x=test$YearBuilt, y=test$HeatingQC)
plot(y=train$SalePrice, x=train$HeatingQC)
### Nothing here

histogram(train$CentralAir)
histogram(test$CentralAir)
plot(x=train$YearBuilt, y=train$CentralAir)
plot(x=test$YearBuilt, y=test$CentralAir)
plot(y=train$SalePrice, x=train$CentralAir)
### Almost all new homes have central air

par(mfrow=c(1,2))
histogram(train$X1stFlrSF)
histogram(test$X1stFlrSF)
plot(x=train$YearBuilt, y=train$X1stFlrSF)
plot(x=test$YearBuilt, y=test$X1stFlrSF)
boxplot(train$X1stFlrSF)
boxplot(test$X1stFlrSF)
plot(y=train$SalePrice, x=train$X1stFlrSF)
### Nothing super evident
### correlated with price
### Remove 3 outliers

histogram(train$X2ndFlrSF)
histogram(test$X2ndFlrSF)
plot(x=train$YearBuilt, y=train$X2ndFlrSF)
plot(x=test$YearBuilt, y=test$X2ndFlrSF)
boxplot(train$X2ndFlrSF)
boxplot(test$X2ndFlrSF)
plot(y=train$SalePrice, x=train$X2ndFlrSF)
### Can generate 2 predictors: bangalow, TotFlrSF

plot(x=train$YearBuilt, y=train$LowQualFinSF)
plot(x=test$YearBuilt, y=test$LowQualFinSF)
boxplot(train$LowQualFinSF)
boxplot(test$LowQualFinSF)
plot(y=train$SalePrice, x=train$LowQualFinSF)
### Not sure what this tells me...

plot(x=train$YearBuilt, y=train$GrLivArea)
plot(x=test$YearBuilt, y=test$GrLivArea)
boxplot(train$GrLivArea)
boxplot(test$GrLivArea)
plot(y=train$SalePrice, x=train$GrLivArea)
### Valid areas
### high correlation

histogram(train$FullBath)
histogram(test$FullBath)
plot(x=train$YearBuilt, y=train$FullBath)
plot(x=test$YearBuilt, y=test$FullBath)
plot(y=train$SalePrice, x=train$FullBath)
### Nothing too bad here

par(mfrow=c(1,2))
histogram(train$HalfBath)
histogram(test$HalfBath)
plot(x=train$YearBuilt, y=train$HalfBath)
plot(x=test$YearBuilt, y=test$HalfBath)
plot(y=train$SalePrice, x=train$HalfBath)
### Nothing

histogram(train$BedroomAbvGr)
histogram(test$BedroomAbvGr)
plot(x=train$YearBuilt, y=train$BedroomAbvGr)
plot(x=test$YearBuilt, y=test$BedroomAbvGr)
plot(y=train$SalePrice, x=train$BedroomAbvGr)
### Nothing

histogram(train$KitchenAbvGr)
histogram(test$KitchenAbvGr)
plot(x=train$YearBuilt, y=train$KitchenAbvGr)
plot(x=test$YearBuilt, y=test$KitchenAbvGr)
plot(y=train$SalePrice, x=train$KitchenAbvGr)
### remove the 3 kitchen in training
### one in test has 0 kitchen
### one in train has 0 kitchen

histogram(train$TotRmsAbvGrd)
histogram(test$TotRmsAbvGrd)
plot(x=train$YearBuilt, y=train$TotRmsAbvGrd)
plot(x=test$YearBuilt, y=test$TotRmsAbvGrd)
plot(y=train$SalePrice, x=train$TotRmsAbvGrd)
### Correlation with rooms

histogram(train$Fireplaces)
histogram(test$Fireplaces)
plot(x=train$YearBuilt, y=train$Fireplaces)
plot(x=test$YearBuilt, y=test$Fireplaces)
plot(y=train$SalePrice, x=train$Fireplaces)
### Slight correlation with price

histogram(train$PavedDrive)
histogram(test$PavedDrive)
plot(x=train$YearBuilt, y=train$PavedDrive)
plot(x=test$YearBuilt, y=test$PavedDrive)
plot(y=train$SalePrice, x=train$PavedDrive)
### Slight correlation with price

histogram(train$WoodDeckSF)
histogram(test$WoodDeckSF)
plot(x=train$YearBuilt, y=train$WoodDeckSF)
plot(x=test$YearBuilt, y=test$WoodDeckSF)
plot(y=train$SalePrice, x=train$WoodDeckSF)
### Maybe create wooddeck predictor?

histogram(train$OpenPorchSF)
histogram(test$OpenPorchSF)
plot(x=train$YearBuilt, y=train$OpenPorchSF)
plot(x=test$YearBuilt, y=test$OpenPorchSF)
plot(y=train$SalePrice, x=train$OpenPorchSF)
### Maybe create openporch predictor?

histogram(train$EnclosedPorch)
histogram(test$EnclosedPorch)
plot(x=train$YearBuilt, y=train$EnclosedPorch)
plot(x=test$YearBuilt, y=test$EnclosedPorch)
plot(y=train$SalePrice, x=train$EnclosedPorch)
### Maybe create enclosedporch predictor?

histogram(train$X3SsnPorch)
histogram(test$X3SsnPorch)
plot(x=train$YearBuilt, y=train$X3SsnPorch)
plot(x=test$YearBuilt, y=test$X3SsnPorch)
plot(y=train$SalePrice, x=train$X3SsnPorch)
### Maybe create 3ssnPorch predictor?

histogram(train$ScreenPorch)
histogram(test$ScreenPorch)
plot(x=train$YearBuilt, y=train$ScreenPorch)
plot(x=test$YearBuilt, y=test$ScreenPorch)
plot(y=train$SalePrice, x=train$ScreenPorch)
### Having screen porch?

### Porch???

histogram(train$PoolArea)
histogram(test$PoolArea)
plot(x=train$YearBuilt, y=train$PoolArea)
plot(x=test$YearBuilt, y=test$PoolArea)
plot(y=train$SalePrice, x=train$PoolArea)
### Having Pool

histogram(train$MiscVal)
histogram(test$MiscVal)
plot(x=train$YearBuilt, y=train$MiscVal)
plot(x=test$YearBuilt, y=test$MiscVal)
plot(y=train$SalePrice, x=train$MiscVal)
### Having Misc

histogram(train$MoSold)
histogram(test$MoSold)
plot(y=train$SalePrice, x=train$MoSold)
### Definitely important

histogram(train$YrSold)
histogram(test$YrSold)
plot(x=train$YrSold, y=train$SalePrice)
### Kind of makes a difference

histogram(train$SaleCondition)
histogram(test$SaleCondition)
plot(x=train$YearBuilt, y=train$SaleCondition)
plot(x=test$YearBuilt, y=test$SaleCondition)
plot(x=train$SaleCondition, y=train$SalePrice)
### All Over the place

histogram(train$Exterior1st)
histogram(test$Exterior1st)
plot(x=train$YearBuilt, y=train$Exterior1st)
plot(x=test$YearBuilt, y=test$Exterior1st)
plot(y=train$SalePrice, x=train$Exterior1st)
### All over the place

histogram(train$Exterior2nd)
histogram(test$Exterior2nd)
plot(x=train$YearBuilt, y=train$Exterior2nd)
plot(x=test$YearBuilt, y=test$Exterior2nd)
plot(y=train$SalePrice, x=train$Exterior2nd)
### All over the place

histogram(train$BsmtFinSF1)
histogram(test$BsmtFinSF1)
plot(x=train$YearBuilt, y=train$BsmtFinSF1)
plot(x=test$YearBuilt, y=test$BsmtFinSF1)
plot(y=train$SalePrice, x=train$BsmtFinSF1)
### Create a new combined BsmtFin
histogram(train$BsmtFinSF2)
histogram(test$BsmtFinSF2)
plot(x=train$YearBuilt, y=train$BsmtFinSF2)
plot(x=test$YearBuilt, y=test$BsmtFinSF2)
plot(y=train$SalePrice, x=train$BsmtFinSF2)

histogram(train$BsmtUnfSF)
histogram(test$BsmtUnfSF)
plot(x=train$YearBuilt, y=train$BsmtUnfSF)
plot(x=test$YearBuilt, y=test$BsmtUnfSF)
plot(y=train$SalePrice, x=train$BsmtUnfSF)
### All over the place, kind of a correlation

histogram(train$TotalBsmtSF)
histogram(test$TotalBsmtSF)
plot(x=train$YearBuilt, y=train$TotalBsmtSF)
plot(x=test$YearBuilt, y=test$TotalBsmtSF)
plot(y=train$SalePrice, x=train$TotalBsmtSF)
### Relationship with SalePrice

histogram(train$Electrical)
histogram(test$Electrical)
plot(x=train$YearBuilt, y=train$Electrical)
plot(x=test$YearBuilt, y=test$Electrical)
plot(y=train$SalePrice, x=train$Electrical)
### Mostly SBrkr

histogram(train$KitchenQual)
histogram(test$KitchenQual)
plot(x=train$YearBuilt, y=train$KitchenQual)
plot(x=test$YearBuilt, y=test$KitchenQual)
plot(y=train$SalePrice, x=train$KitchenQual)
### Nothing here

histogram(train$GarageCars)
histogram(test$GarageCars)
plot(x=train$YearBuilt, y=train$GarageCars)
plot(x=test$YearBuilt, y=test$GarageCars)
plot(y=train$SalePrice, x=train$GarageCars)
# Trend with price

histogram(train$GarageArea)
histogram(test$GarageArea)
plot(x=train$YearBuilt, y=train$GarageArea)
plot(x=test$YearBuilt, y=test$GarageArea)
plot(y=train$SalePrice, x=train$GarageArea)
# Trend with price

histogram(train$SaleType)
histogram(test$SaleType)
plot(x=train$YearBuilt, y=train$SaleType)
plot(x=test$YearBuilt, y=test$SaleType)
plot(y=train$SalePrice, x=train$SaleType)
### Mostly WD

histogram(train$Utilities)
histogram(test$Utilities)
plot(x=train$YearBuilt, y=train$Utilities)
plot(x=test$YearBuilt, y=test$Utilities)
plot(y=train$SalePrice, x=train$Utilities)
### remove untilities outlier in training

histogram(train$BsmtFullBath)
histogram(test$BsmtFullBath)
plot(x=train$YearBuilt, y=train$BsmtFullBath)
plot(x=test$YearBuilt, y=test$BsmtFullBath)
plot(y=train$SalePrice, x=train$BsmtFullBath)
### Vague trend

histogram(train$BsmtHalfBath)
histogram(test$BsmtHalfBath)
plot(x=train$YearBuilt, y=train$BsmtHalfBath)
plot(x=test$YearBuilt, y=test$BsmtHalfBath)
plot(y=train$SalePrice, x=train$BsmtHalfBath)
### Vague trend

histogram(train$Functional)
histogram(test$Functional)
plot(x=train$YearBuilt, y=train$Functional)
plot(x=test$YearBuilt, y=test$Functional)
plot(y=train$SalePrice, x=train$Functional)
### Mostly Typ

histogram(train$MSZoning)
histogram(test$MSZoning)
plot(x=train$YearBuilt, y=train$MSZoning)
plot(x=test$YearBuilt, y=test$MSZoning)
plot(y=train$SalePrice, x=train$MSZoning)

histogram(train$MasVnrArea)
histogram(test$MasVnrArea)
plot(x=train$YearBuilt, y=train$MasVnrArea)
plot(x=test$YearBuilt, y=test$MasVnrArea)
plot(y=train$SalePrice, x=train$MasVnrArea)
### Trend with price
### A lot of zeros

histogram(train$MasVnrType)
histogram(test$MasVnrType)
plot(x=train$YearBuilt, y=train$MasVnrType)
plot(x=test$YearBuilt, y=test$MasVnrType)
plot(y=train$SalePrice, x=train$MasVnrType)

histogram(train$BsmtFinType1)
histogram(test$BsmtFinType1)
plot(x=train$YearBuilt, y=train$BsmtFinType1)
plot(x=test$YearBuilt, y=test$BsmtFinType1)
plot(y=train$SalePrice, x=train$BsmtFinType1)

histogram(train$BsmtFinType2)
histogram(test$BsmtFinType2)
plot(x=train$YearBuilt, y=train$BsmtFinType2)
plot(x=test$YearBuilt, y=test$BsmtFinType2)
plot(y=train$SalePrice, x=train$BsmtFinType2)
### Mostly Unf

histogram(train$BsmtQual)
histogram(test$BsmtQual)
plot(x=train$YearBuilt, y=train$BsmtQual)
plot(x=test$YearBuilt, y=test$BsmtQual)
plot(y=train$SalePrice, x=train$BsmtQual)

histogram(train$BsmtCond)
histogram(test$BsmtCond)
plot(x=train$YearBuilt, y=train$BsmtCond)
plot(x=test$YearBuilt, y=test$BsmtCond)
plot(y=train$SalePrice, x=train$BsmtCond)
### Mostly TA

histogram(train$BsmtExposure)
histogram(test$BsmtExposure)
plot(x=train$YearBuilt, y=train$BsmtExposure)
plot(x=test$YearBuilt, y=test$BsmtExposure)
plot(y=train$SalePrice, x=train$BsmtExposure)

histogram(train$GarageType)
histogram(test$GarageType)
plot(x=train$YearBuilt, y=train$GarageType)
plot(x=test$YearBuilt, y=test$GarageType)
plot(y=train$SalePrice, x=train$GarageType)

histogram(train$GarageYrBlt)
histogram(test$GarageYrBlt)
plot(x=train$YearBuilt, y=train$GarageYrBlt)
plot(x=test$YearBuilt, y=test$GarageYrBlt)
plot(y=train$SalePrice, x=train$GarageYrBlt)
sum(dataFinal$YearBuilt ==dataFinal$GarageYrBlt)
### Outlier in test

histogram(train$GarageFinish)
histogram(test$GarageFinish)
plot(x=train$YearBuilt, y=train$GarageFinish)
plot(x=test$YearBuilt, y=test$GarageFinish)
plot(y=train$SalePrice, x=train$GarageFinish)

histogram(train$GarageQual)
histogram(test$GarageQual)
plot(x=train$YearBuilt, y=train$GarageQual)
plot(x=test$YearBuilt, y=test$GarageQual)
plot(y=train$SalePrice, x=train$GarageQual)

histogram(train$GarageCond)
histogram(test$GarageCond)
plot(x=train$YearBuilt, y=train$GarageCond)
plot(x=test$YearBuilt, y=test$GarageCond)
plot(y=train$SalePrice, x=train$GarageCond)

histogram(train$LotFrontage)
histogram(test$LotFrontage)
plot(x=train$YearBuilt, y=train$LotFrontage)
plot(x=test$YearBuilt, y=test$LotFrontage)
plot(y=train$SalePrice, x=train$LotFrontage)

histogram(train$FireplaceQu)
histogram(test$FireplaceQu)
plot(x=train$YearBuilt, y=train$FireplaceQu)
plot(x=test$YearBuilt, y=test$FireplaceQu)
plot(y=train$SalePrice, x=train$FireplaceQu)
### Having fireplaces

histogram(train$Fence)
histogram(test$Fence)
plot(x=train$YearBuilt, y=train$Fence)
plot(x=test$YearBuilt, y=test$Fence)
plot(y=train$SalePrice, x=train$Fence)
### having a fence makes a difference?

histogram(train$Alley)
histogram(test$Alley)
plot(x=train$YearBuilt, y=train$Alley)
plot(x=test$YearBuilt, y=test$Alley)
plot(y=train$SalePrice, x=train$Alley)
### having an alley makes a difference?

histogram(train$MiscFeature)
histogram(test$MiscFeature)
plot(x=train$YearBuilt, y=train$MiscFeature)
plot(x=test$YearBuilt, y=test$MiscFeature)
plot(y=train$SalePrice, x=train$MiscFeature)
### having misc features makes a difference?

histogram(train$PoolQC)
histogram(test$PoolQC)
plot(x=train$YearBuilt, y=train$PoolQC)
plot(x=test$YearBuilt, y=test$PoolQC)
plot(y=train$SalePrice, x=train$PoolQC)
### Having a pool makes a difference?

