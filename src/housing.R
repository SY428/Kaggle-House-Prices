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

# Get files
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Set as appropriate (factor, num, etc)
train$SalePrice <- as.numeric(train$SalePrice)

#train$ <- as.numeric(train$)
#train$ <- as.factor(train$)

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


# Add test "SalePrice"
test <- data.frame(test, SalePrice = as.integer(rep(0,nrow(test))))

# Generate indexing to assign back to train and test later
train_id <- rep(TRUE, nrow(train))
test_id <- rep(FALSE, nrow(test))

dataFullA <- rbind(train,test)
dataFull <- data.frame(dataFullA[(names(dataFullA) != "SalePrice")], is_train = c(train_id,test_id))

# Add index to each observation if needed
#data_index <- 1:nrow(dataFull)
#dataFull <- data.frame(dataFull, data_index = data_index)

# Check all non values
apply(dataFull,2,function(x) sum(x == ""))

# Set empty values to NAs
dataFull[dataFull == ""] <- NA

# Check all NAs in 
apply(dataFull,2,function(x) sum(is.na(x)))

#############
# PREDICTIONS
#############

# Predict NA values of the following, from smallest to largest
temp <- which(apply(dataFull,2,function(x) sum(is.na(x))) > 0)
temp1 <- apply(dataFull[temp],2,function(x) sum(is.na(x)))
namesNA <- names(sort(temp1)) # Ordered by number of 

dataFull_nonNA <- dataFull[!(names(dataFull) %in% namesNA)]
dataFull_NA <- dataFull[names(dataFull) %in% namesNA]

#########################################
### Exterior1st, factors --> randomForest
dataExt1 <- data.frame(dataFull_nonNA, Exterior1st = dataFull_NA$Exterior1st)
Ext1_nonNA <- droplevels(subset(dataExt1, !is.na(dataExt1$Exterior1st)))
Ext1_NA <- subset(dataExt1, is.na(dataExt1$Exterior1st))
Ext1Model <- randomForest(as.factor(Exterior1st) ~  ., data = Ext1_nonNA[(names(Ext1_nonNA) != "is_train") & (names(Ext1_nonNA) != "Id")])

Ext1_new <- predict(Ext1Model, newdata = data.frame(Ext1_NA[names(Ext1_NA) != "Exterior1st"]))
Ext1_NA[is.na(Ext1_NA$Exterior1st)]$Exterior1st <- Ext1_new
dataExt1 <- rbind(Ext1_nonNA, Ext1_NA)

dataFull_nonNA <- dataExt1
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "Exterior1st"]

#########################################
### Exterior2nd, factors --> randomForest
dataExt2 <- data.frame(dataFull_nonNA, Exterior2nd = dataFull_NA$Exterior2nd)
Ext2_nonNA <- droplevels(subset(dataExt2, !is.na(dataExt2$Exterior2nd)))
Ext2_NA <- subset(dataExt2, is.na(dataExt2$Exterior2nd))
Ext2Model <- randomForest(as.factor(Exterior2nd) ~  ., data = Ext2_nonNA[(names(Ext2_nonNA) != "is_train") & (names(Ext2_nonNA) != "Id")])

Ext2_new <- predict(Ext2Model, newdata = data.frame(Ext2_NA[names(Ext2_NA) != "Exterior2nd"]))
Ext2_NA[is.na(Ext2_NA$Exterior2nd)]$Exterior2nd <- Ext2_new
dataExt2 <- rbind(Ext2_nonNA, Ext2_NA)

dataFull_nonNA <- dataExt2
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "Exterior2nd"]

#############################################
### BsmtFinSF1, numeric --> linear regression
dataBsmtFinSF1 <- data.frame(dataFull_nonNA, BsmtFinSF1 = dataFull_NA$BsmtFinSF1)
BsmtFinSF1_nonNA <- subset(dataBsmtFinSF1, !is.na(dataBsmtFinSF1$BsmtFinSF1))
BsmtFinSF1_NA <- subset(dataBsmtFinSF1, is.na(dataBsmtFinSF1$BsmtFinSF1))

fitBsmtFinSF1 <- lm(BsmtFinSF1 ~ . -1, data = BsmtFinSF1_nonNA[(names(BsmtFinSF1_nonNA) != "is_train") & (names(BsmtFinSF1_nonNA) != "Id")])
my_anova <- anova(fitBsmtFinSF1)
BsmtFinSF1_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
BsmtFinSF1Model <- lm(BsmtFinSF1 ~ ., data = BsmtFinSF1_nonNA[(names(BsmtFinSF1_nonNA) != "is_train") & (names(BsmtFinSF1_nonNA) != "Id") & ((names(BsmtFinSF1_nonNA) == "BsmtFinSF1") | (names(BsmtFinSF1_nonNA) %in% BsmtFinSF1_features))])

BsmtFinSF1_new <- predict(BsmtFinSF1Model, newdata = data.frame(BsmtFinSF1_NA[names(BsmtFinSF1_NA) != "BsmtFinSF1"]))
BsmtFinSF1_NA[is.na(BsmtFinSF1_NA$BsmtFinSF1)]$BsmtFinSF1 <- BsmtFinSF1_new
dataBsmtFinSF1 <- rbind(BsmtFinSF1_nonNA, BsmtFinSF1_NA)

dataFull_nonNA <- dataBsmtFinSF1
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "BsmtFinSF1"]

#############################################
###"BsmtFinSF2", numeric --> linear regression
dataBsmtFinSF2 <- data.frame(dataFull_nonNA, BsmtFinSF2 = dataFull_NA$BsmtFinSF2)
BsmtFinSF2_nonNA <- subset(dataBsmtFinSF2, !is.na(dataBsmtFinSF2$BsmtFinSF2))
BsmtFinSF2_NA <- subset(dataBsmtFinSF2, is.na(dataBsmtFinSF2$BsmtFinSF2))

fitBsmtFinSF2 <- lm(BsmtFinSF2 ~ . -1, data = BsmtFinSF2_nonNA[(names(BsmtFinSF2_nonNA) != "is_train") & (names(BsmtFinSF2_nonNA) != "Id")])
my_anova <- anova(fitBsmtFinSF2)
BsmtFinSF2_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
BsmtFinSF2Model <- lm(BsmtFinSF2 ~ ., data = BsmtFinSF2_nonNA[(names(BsmtFinSF2_nonNA) != "is_train") & (names(BsmtFinSF2_nonNA) != "Id") & ((names(BsmtFinSF2_nonNA) == "BsmtFinSF2") | (names(BsmtFinSF2_nonNA) %in% BsmtFinSF2_features))])

BsmtFinSF2_new <- predict(BsmtFinSF2Model, newdata = data.frame(BsmtFinSF2_NA[names(BsmtFinSF2_NA) != "BsmtFinSF2"]))
BsmtFinSF2_NA[is.na(BsmtFinSF2_NA$BsmtFinSF2)]$BsmtFinSF2 <- BsmtFinSF2_new
dataBsmtFinSF2 <- rbind(BsmtFinSF2_nonNA, BsmtFinSF2_NA)

dataFull_nonNA <- dataBsmtFinSF2
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "BsmtFinSF2"]

#############################################
###"BsmtUnfSF", numeric --> linear regression
dataBsmtUnfSF <- data.frame(dataFull_nonNA, BsmtUnfSF = dataFull_NA$BsmtUnfSF)
BsmtUnfSF_nonNA <- subset(dataBsmtUnfSF, !is.na(dataBsmtUnfSF$BsmtUnfSF))
BsmtUnfSF_NA <- subset(dataBsmtUnfSF, is.na(dataBsmtUnfSF$BsmtUnfSF))

fitBsmtUnfSF <- lm(BsmtUnfSF ~ . -1, data = BsmtUnfSF_nonNA[(names(BsmtUnfSF_nonNA) != "is_train") & (names(BsmtUnfSF_nonNA) != "Id")])
my_anova <- anova(fitBsmtUnfSF)
BsmtUnfSF_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
BsmtUnfSFModel <- lm(BsmtUnfSF ~ ., data = BsmtUnfSF_nonNA[(names(BsmtUnfSF_nonNA) != "is_train") & (names(BsmtUnfSF_nonNA) != "Id") & ((names(BsmtUnfSF_nonNA) == "BsmtUnfSF") | (names(BsmtUnfSF_nonNA) %in% BsmtUnfSF_features))])

BsmtUnfSF_new <- predict(BsmtUnfSFModel, newdata = data.frame(BsmtUnfSF_NA[names(BsmtUnfSF_NA) != "BsmtUnfSF"]))
BsmtUnfSF_NA[is.na(BsmtUnfSF_NA$BsmtUnfSF)]$BsmtUnfSF <- BsmtUnfSF_new
dataBsmtUnfSF <- rbind(BsmtUnfSF_nonNA, BsmtUnfSF_NA)

dataFull_nonNA <- dataBsmtUnfSF
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "BsmtUnfSF"]

#################################################
###"TotalBsmtSF", numeric --> linear regression
dataTotalBsmtSF <- data.frame(dataFull_nonNA, TotalBsmtSF = dataFull_NA$TotalBsmtSF)
TotalBsmtSF_nonNA <- subset(dataTotalBsmtSF, !is.na(dataTotalBsmtSF$TotalBsmtSF))
TotalBsmtSF_NA <- subset(dataTotalBsmtSF, is.na(dataTotalBsmtSF$TotalBsmtSF))

fitTotalBsmtSF <- lm(TotalBsmtSF ~ . -1, data = TotalBsmtSF_nonNA[(names(TotalBsmtSF_nonNA) != "is_train") & (names(TotalBsmtSF_nonNA) != "Id")])
my_anova <- anova(fitTotalBsmtSF)
TotalBsmtSF_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
TotalBsmtSFModel <- lm(TotalBsmtSF ~ ., data = TotalBsmtSF_nonNA[(names(TotalBsmtSF_nonNA) != "is_train") & (names(TotalBsmtSF_nonNA) != "Id") & ((names(TotalBsmtSF_nonNA) == "TotalBsmtSF") | (names(TotalBsmtSF_nonNA) %in% TotalBsmtSF_features))])

TotalBsmtSF_new <- predict(TotalBsmtSFModel, newdata = data.frame(TotalBsmtSF_NA[names(TotalBsmtSF_NA) != "TotalBsmtSF"]))
TotalBsmtSF_NA[is.na(TotalBsmtSF_NA$TotalBsmtSF)]$TotalBsmtSF <- TotalBsmtSF_new
dataTotalBsmtSF <- rbind(TotalBsmtSF_nonNA, TotalBsmtSF_NA)

dataFull_nonNA <- dataTotalBsmtSF
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "TotalBsmtSF"]

#################################################
###"Electrical", factors --> randomForest
dataElectrical <- data.frame(dataFull_nonNA, Electrical = dataFull_NA$Electrical)
Electrical_nonNA <- droplevels(subset(dataElectrical, !is.na(dataElectrical$Electrical)))
Electrical_NA <- subset(dataElectrical, is.na(dataElectrical$Electrical))
ElectricalModel <- randomForest(as.factor(Electrical) ~  ., data = Electrical_nonNA[(names(Electrical_nonNA) != "is_train") & (names(Electrical_nonNA) != "Id")])

Electrical_new <- predict(ElectricalModel, newdata = data.frame(Electrical_NA[names(Electrical_NA) != "Electrical"]))
Electrical_NA[is.na(Electrical_NA$Electrical)]$Electrical <- Electrical_new
dataElectrical <- rbind(Electrical_nonNA, Electrical_NA)

dataFull_nonNA <- dataElectrical
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "Electrical"]

#################################################
###"KitchenQual", factors --> randomForest
dataKitchenQual <- data.frame(dataFull_nonNA, KitchenQual = dataFull_NA$KitchenQual)
KitchenQual_nonNA <- droplevels(subset(dataKitchenQual, !is.na(dataKitchenQual$KitchenQual)))
KitchenQual_NA <- subset(dataKitchenQual, is.na(dataKitchenQual$KitchenQual))
KitchenQualModel <- randomForest(as.factor(KitchenQual) ~  ., data = KitchenQual_nonNA[(names(KitchenQual_nonNA) != "is_train") & (names(KitchenQual_nonNA) != "Id")])

KitchenQual_new <- predict(KitchenQualModel, newdata = data.frame(KitchenQual_NA[names(KitchenQual_NA) != "KitchenQual"]))
KitchenQual_NA[is.na(KitchenQual_NA$KitchenQual)]$KitchenQual <- KitchenQual_new
dataKitchenQual <- rbind(KitchenQual_nonNA, KitchenQual_NA)

dataFull_nonNA <- dataKitchenQual
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "KitchenQual"]

#################################################
###"GarageCars", numeric --> linear regression 
### XXXXX factors --> randomForest XXXXX
dataGarageCars <- data.frame(dataFull_nonNA, GarageCars = dataFull_NA$GarageCars)
GarageCars_nonNA <- droplevels(subset(dataGarageCars, !is.na(dataGarageCars$GarageCars)))
GarageCars_NA <- subset(dataGarageCars, is.na(dataGarageCars$GarageCars))

###GarageCarsModel <- randomForest(as.factor(GarageCars) ~  ., data = GarageCars_nonNA[(names(GarageCars_nonNA) != "is_train") & (names(GarageCars_nonNA) != "Id")])

fitGarageCars <- lm(GarageCars ~ . -1, data = GarageCars_nonNA[(names(GarageCars_nonNA) != "is_train") & (names(GarageCars_nonNA) != "Id")])
my_anova <- anova(fitGarageCars)
GarageCars_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
GarageCarsModel <- lm(GarageCars ~ ., data = GarageCars_nonNA[(names(GarageCars_nonNA) != "is_train") & (names(GarageCars_nonNA) != "Id") & ((names(GarageCars_nonNA) == "GarageCars") | (names(GarageCars_nonNA) %in% GarageCars_features))])

GarageCars_new <- predict(GarageCarsModel, newdata = data.frame(GarageCars_NA[names(GarageCars_NA) != "GarageCars"]))
GarageCars_NA[is.na(GarageCars_NA$GarageCars)]$GarageCars <- GarageCars_new
dataGarageCars <- rbind(GarageCars_nonNA, GarageCars_NA)

dataFull_nonNA <- dataGarageCars
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "GarageCars"]

#################################################
###"GarageArea", numeric --> linear regression
dataGarageArea <- data.frame(dataFull_nonNA, GarageArea = dataFull_NA$GarageArea)
GarageArea_nonNA <- subset(dataGarageArea, !is.na(dataGarageArea$GarageArea))
GarageArea_NA <- subset(dataGarageArea, is.na(dataGarageArea$GarageArea))

fitGarageArea <- lm(GarageArea ~ . -1, data = GarageArea_nonNA[(names(GarageArea_nonNA) != "is_train") & (names(GarageArea_nonNA) != "Id")])
my_anova <- anova(fitGarageArea)
GarageArea_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
GarageAreaModel <- lm(GarageArea ~ ., data = GarageArea_nonNA[(names(GarageArea_nonNA) != "is_train") & (names(GarageArea_nonNA) != "Id") & ((names(GarageArea_nonNA) == "GarageArea") | (names(GarageArea_nonNA) %in% GarageArea_features))])

GarageArea_new <- predict(GarageAreaModel, newdata = data.frame(GarageArea_NA[names(GarageArea_NA) != "GarageArea"]))
GarageArea_NA[is.na(GarageArea_NA$GarageArea)]$GarageArea <- GarageArea_new
dataGarageArea <- rbind(GarageArea_nonNA, GarageArea_NA)

dataFull_nonNA <- dataGarageArea
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "GarageArea"]

#################################################
###"SaleType", factors --> randomForest
dataSaleType <- data.frame(dataFull_nonNA, SaleType = dataFull_NA$SaleType)
SaleType_nonNA <- droplevels(subset(dataSaleType, !is.na(dataSaleType$SaleType)))
SaleType_NA <- subset(dataSaleType, is.na(dataSaleType$SaleType))
SaleTypeModel <- randomForest(as.factor(SaleType) ~  ., data = SaleType_nonNA[(names(SaleType_nonNA) != "is_train") & (names(SaleType_nonNA) != "Id")])

SaleType_new <- predict(SaleTypeModel, newdata = data.frame(SaleType_NA[names(SaleType_NA) != "SaleType"]))
SaleType_NA[is.na(SaleType_NA$SaleType)]$SaleType <- SaleType_new
dataSaleType <- rbind(SaleType_nonNA, SaleType_NA)

dataFull_nonNA <- dataSaleType
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "SaleType"]

#################################################
###"Utilities", factors --> randomForest
dataUtilities <- data.frame(dataFull_nonNA, Utilities = dataFull_NA$Utilities)
Utilities_nonNA <- droplevels(subset(dataUtilities, !is.na(dataUtilities$Utilities)))
Utilities_NA <- subset(dataUtilities, is.na(dataUtilities$Utilities))
UtilitiesModel <- randomForest(as.factor(Utilities) ~  ., data = Utilities_nonNA[(names(Utilities_nonNA) != "is_train") & (names(Utilities_nonNA) != "Id")])

Utilities_new <- predict(UtilitiesModel, newdata = data.frame(Utilities_NA[names(Utilities_NA) != "Utilities"]))
Utilities_NA[is.na(Utilities_NA$Utilities)]$Utilities <- Utilities_new
dataUtilities <- rbind(Utilities_nonNA, Utilities_NA)

dataFull_nonNA <- dataUtilities
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "Utilities"]

#################################################
###"BsmtFullBath", numeric --> linear regression
### XXXX factors --> randomForest XXXX
dataBsmtFullBath <- data.frame(dataFull_nonNA, BsmtFullBath = dataFull_NA$BsmtFullBath)
BsmtFullBath_nonNA <- droplevels(subset(dataBsmtFullBath, !is.na(dataBsmtFullBath$BsmtFullBath)))
BsmtFullBath_NA <- subset(dataBsmtFullBath, is.na(dataBsmtFullBath$BsmtFullBath))

#BsmtFullBathModel <- randomForest(as.factor(BsmtFullBath) ~  ., data = BsmtFullBath_nonNA[(names(BsmtFullBath_nonNA) != "is_train") & (names(BsmtFullBath_nonNA) != "Id")])

fitBsmtFullBath <- lm(BsmtFullBath ~ . -1, data = BsmtFullBath_nonNA[(names(BsmtFullBath_nonNA) != "is_train") & (names(BsmtFullBath_nonNA) != "Id")])
my_anova <- anova(fitBsmtFullBath)
BsmtFullBath_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
BsmtFullBathModel <- lm(BsmtFullBath ~ ., data = BsmtFullBath_nonNA[(names(BsmtFullBath_nonNA) != "is_train") & (names(BsmtFullBath_nonNA) != "Id") & ((names(BsmtFullBath_nonNA) == "BsmtFullBath") | (names(BsmtFullBath_nonNA) %in% BsmtFullBath_features))])

BsmtFullBath_new <- predict(BsmtFullBathModel, newdata = data.frame(BsmtFullBath_NA[names(BsmtFullBath_NA) != "BsmtFullBath"]))
BsmtFullBath_NA[is.na(BsmtFullBath_NA$BsmtFullBath)]$BsmtFullBath <- BsmtFullBath_new
dataBsmtFullBath <- rbind(BsmtFullBath_nonNA, BsmtFullBath_NA)

dataFull_nonNA <- dataBsmtFullBath
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "BsmtFullBath"]

#################################################
###"BsmtHalfBath", numeric --> linear regression
### XXXX factors --> randomForest XXXX
dataBsmtHalfBath <- data.frame(dataFull_nonNA, BsmtHalfBath = dataFull_NA$BsmtHalfBath)
BsmtHalfBath_nonNA <- droplevels(subset(dataBsmtHalfBath, !is.na(dataBsmtHalfBath$BsmtHalfBath)))
BsmtHalfBath_NA <- subset(dataBsmtHalfBath, is.na(dataBsmtHalfBath$BsmtHalfBath))

# BsmtHalfBathModel <- randomForest(as.factor(BsmtHalfBath) ~  ., data = BsmtHalfBath_nonNA[(names(BsmtHalfBath_nonNA) != "is_train") & (names(BsmtHalfBath_nonNA) != "Id")])

fitBsmtHalfBath <- lm(BsmtHalfBath ~ . -1, data = BsmtHalfBath_nonNA[(names(BsmtHalfBath_nonNA) != "is_train") & (names(BsmtHalfBath_nonNA) != "Id")])
my_anova <- anova(fitBsmtHalfBath)
BsmtHalfBath_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
BsmtHalfBathModel <- lm(BsmtHalfBath ~ ., data = BsmtHalfBath_nonNA[(names(BsmtHalfBath_nonNA) != "is_train") & (names(BsmtHalfBath_nonNA) != "Id") & ((names(BsmtHalfBath_nonNA) == "BsmtHalfBath") | (names(BsmtHalfBath_nonNA) %in% BsmtHalfBath_features))])

BsmtHalfBath_new <- predict(BsmtHalfBathModel, newdata = data.frame(BsmtHalfBath_NA[names(BsmtHalfBath_NA) != "BsmtHalfBath"]))
BsmtHalfBath_NA[is.na(BsmtHalfBath_NA$BsmtHalfBath)]$BsmtHalfBath <- BsmtHalfBath_new
dataBsmtHalfBath <- rbind(BsmtHalfBath_nonNA, BsmtHalfBath_NA)

dataFull_nonNA <- dataBsmtHalfBath
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "BsmtHalfBath"]

#################################################
###"Functional", factors --> randomForest
dataFunctional <- data.frame(dataFull_nonNA, Functional = dataFull_NA$Functional)
Functional_nonNA <- droplevels(subset(dataFunctional, !is.na(dataFunctional$Functional)))
Functional_NA <- subset(dataFunctional, is.na(dataFunctional$Functional))
FunctionalModel <- randomForest(as.factor(Functional) ~  ., data = Functional_nonNA[(names(Functional_nonNA) != "is_train") & (names(Functional_nonNA) != "Id")])

Functional_new <- predict(FunctionalModel, newdata = data.frame(Functional_NA[names(Functional_NA) != "Functional"]))
Functional_NA[is.na(Functional_NA$Functional)]$Functional <- Functional_new
dataFunctional <- rbind(Functional_nonNA, Functional_NA)

dataFull_nonNA <- dataFunctional
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "Functional"]

#################################################
###"MSZoning", factors --> randomForest
dataMSZoning <- data.frame(dataFull_nonNA, MSZoning = dataFull_NA$MSZoning)
MSZoning_nonNA <- droplevels(subset(dataMSZoning, !is.na(dataMSZoning$MSZoning)))
MSZoning_NA <- subset(dataMSZoning, is.na(dataMSZoning$MSZoning))
MSZoningModel <- randomForest(as.factor(MSZoning) ~  ., data = MSZoning_nonNA[(names(MSZoning_nonNA) != "is_train") & (names(MSZoning_nonNA) != "Id")])

MSZoning_new <- predict(MSZoningModel, newdata = data.frame(MSZoning_NA[names(MSZoning_NA) != "MSZoning"]))
MSZoning_NA[is.na(MSZoning_NA$MSZoning)]$MSZoning <- MSZoning_new
dataMSZoning <- rbind(MSZoning_nonNA, MSZoning_NA)

dataFull_nonNA <- dataMSZoning
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "MSZoning"]

#################################################
###"MasVnrArea", numeric --> linear regression
dataMasVnrArea <- data.frame(dataFull_nonNA, MasVnrArea = dataFull_NA$MasVnrArea)
MasVnrArea_nonNA <- subset(dataMasVnrArea, !is.na(dataMasVnrArea$MasVnrArea))
MasVnrArea_NA <- subset(dataMasVnrArea, is.na(dataMasVnrArea$MasVnrArea))

fitMasVnrArea <- lm(MasVnrArea ~ . -1, data = MasVnrArea_nonNA[(names(MasVnrArea_nonNA) != "is_train") & (names(MasVnrArea_nonNA) != "Id")])
my_anova <- anova(fitMasVnrArea)
MasVnrArea_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
MasVnrAreaModel <- lm(MasVnrArea ~ ., data = MasVnrArea_nonNA[(names(MasVnrArea_nonNA) != "is_train") & (names(MasVnrArea_nonNA) != "Id") & ((names(MasVnrArea_nonNA) == "MasVnrArea") | (names(MasVnrArea_nonNA) %in% MasVnrArea_features))])

MasVnrArea_new <- predict(MasVnrAreaModel, newdata = data.frame(MasVnrArea_NA[names(MasVnrArea_NA) != "MasVnrArea"]))
MasVnrArea_NA[is.na(MasVnrArea_NA$MasVnrArea)]$MasVnrArea <- MasVnrArea_new
dataMasVnrArea <- rbind(MasVnrArea_nonNA, MasVnrArea_NA)

dataFull_nonNA <- dataMasVnrArea
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "MasVnrArea"]

#################################################
###"MasVnrType", factors --> randomForest
dataMasVnrType <- data.frame(dataFull_nonNA, MasVnrType = dataFull_NA$MasVnrType)
MasVnrType_nonNA <- droplevels(subset(dataMasVnrType, !is.na(dataMasVnrType$MasVnrType)))
MasVnrType_NA <- subset(dataMasVnrType, is.na(dataMasVnrType$MasVnrType))
MasVnrTypeModel <- randomForest(as.factor(MasVnrType) ~  ., data = MasVnrType_nonNA[(names(MasVnrType_nonNA) != "is_train") & (names(MasVnrType_nonNA) != "Id")])

MasVnrType_new <- predict(MasVnrTypeModel, newdata = data.frame(MasVnrType_NA[names(MasVnrType_NA) != "MasVnrType"]))
MasVnrType_NA[is.na(MasVnrType_NA$MasVnrType)]$MasVnrType <- MasVnrType_new
dataMasVnrType <- rbind(MasVnrType_nonNA, MasVnrType_NA)

dataFull_nonNA <- dataMasVnrType
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "MasVnrType"]

#################################################
###"BsmtFinType1", factors --> randomForest
dataBsmtFinType1 <- data.frame(dataFull_nonNA, BsmtFinType1 = dataFull_NA$BsmtFinType1)
BsmtFinType1_nonNA <- droplevels(subset(dataBsmtFinType1, !is.na(dataBsmtFinType1$BsmtFinType1)))
BsmtFinType1_NA <- subset(dataBsmtFinType1, is.na(dataBsmtFinType1$BsmtFinType1))
BsmtFinType1Model <- randomForest(as.factor(BsmtFinType1) ~  ., data = BsmtFinType1_nonNA[(names(BsmtFinType1_nonNA) != "is_train") & (names(BsmtFinType1_nonNA) != "Id") & (names(BsmtFinType1_nonNA) != "Condition2")])

BsmtFinType1_new <- predict(BsmtFinType1Model, newdata = data.frame(BsmtFinType1_NA[(names(BsmtFinType1_NA) != "BsmtFinType1") & (names(BsmtFinType1_NA) != "Condition2")]))
BsmtFinType1_NA$BsmtFinType1 <- BsmtFinType1_new
dataBsmtFinType1 <- rbind(BsmtFinType1_nonNA, BsmtFinType1_NA)

dataFull_nonNA <- dataBsmtFinType1
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "BsmtFinType1"]

#################################################
###"BsmtFinType2", factors --> randomForest
dataBsmtFinType2 <- data.frame(dataFull_nonNA, BsmtFinType2 = dataFull_NA$BsmtFinType2)
BsmtFinType2_nonNA <- droplevels(subset(dataBsmtFinType2, !is.na(dataBsmtFinType2$BsmtFinType2)))
BsmtFinType2_NA <- subset(dataBsmtFinType2, is.na(dataBsmtFinType2$BsmtFinType2))
BsmtFinType2Model <- randomForest(as.factor(BsmtFinType2) ~  ., data = BsmtFinType2_nonNA[(names(BsmtFinType2_nonNA) != "is_train") & (names(BsmtFinType2_nonNA) != "Id")])

BsmtFinType2_new <- predict(BsmtFinType2Model, newdata = data.frame(BsmtFinType2_NA[names(BsmtFinType2_NA) != "BsmtFinType2"]))
BsmtFinType2_NA$BsmtFinType2 <- BsmtFinType2_new
dataBsmtFinType2 <- rbind(BsmtFinType2_nonNA, BsmtFinType2_NA)

dataFull_nonNA <- dataBsmtFinType2
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "BsmtFinType2"]

#################################################
###"BsmtQual", factors --> randomForest
dataBsmtQual <- data.frame(dataFull_nonNA, BsmtQual = dataFull_NA$BsmtQual)
BsmtQual_nonNA <- droplevels(subset(dataBsmtQual, !is.na(dataBsmtQual$BsmtQual)))
BsmtQual_NA <- subset(dataBsmtQual, is.na(dataBsmtQual$BsmtQual))
BsmtQualModel <- randomForest(as.factor(BsmtQual) ~  ., data = BsmtQual_nonNA[(names(BsmtQual_nonNA) != "is_train") & (names(BsmtQual_nonNA) != "Id")])

BsmtQual_new <- predict(BsmtQualModel, newdata = data.frame(BsmtQual_NA[names(BsmtQual_NA) != "BsmtQual"]))
BsmtQual_NA$BsmtQual <- BsmtQual_new
dataBsmtQual <- rbind(BsmtQual_nonNA, BsmtQual_NA)

dataFull_nonNA <- dataBsmtQual
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "BsmtQual"]

#################################################
###"BsmtCond", factors --> randomForest
dataBsmtCond <- data.frame(dataFull_nonNA, BsmtCond = dataFull_NA$BsmtCond)
BsmtCond_nonNA <- droplevels(subset(dataBsmtCond, !is.na(dataBsmtCond$BsmtCond)))
BsmtCond_NA <- subset(dataBsmtCond, is.na(dataBsmtCond$BsmtCond))
BsmtCondModel <- randomForest(as.factor(BsmtCond) ~  ., data = BsmtCond_nonNA[(names(BsmtCond_nonNA) != "is_train") & (names(BsmtCond_nonNA) != "Id") & (names(BsmtCond_nonNA) != "RoofMatl") & (names(BsmtCond_nonNA) != "Utilities") & (names(BsmtCond_nonNA) != "Fireplaces")])

BsmtCond_new <- predict(BsmtCondModel, newdata = data.frame(BsmtCond_NA[names(BsmtCond_NA) != "BsmtCond"]))
BsmtCond_NA$BsmtCond <- BsmtCond_new
dataBsmtCond <- rbind(BsmtCond_nonNA, BsmtCond_NA)

dataFull_nonNA <- dataBsmtCond
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "BsmtCond"]

#################################################
###"BsmtExposure", factors --> randomForest
dataBsmtExposure <- data.frame(dataFull_nonNA, BsmtExposure = dataFull_NA$BsmtExposure)
BsmtExposure_nonNA <- droplevels(subset(dataBsmtExposure, !is.na(dataBsmtExposure$BsmtExposure)))
BsmtExposure_NA <- subset(dataBsmtExposure, is.na(dataBsmtExposure$BsmtExposure))
BsmtExposureModel <- randomForest(as.factor(BsmtExposure) ~  ., data = BsmtExposure_nonNA[(names(BsmtExposure_nonNA) != "is_train") & (names(BsmtExposure_nonNA) != "Id") & (names(BsmtExposure_nonNA) != "Condition2")])

BsmtExposure_new <- predict(BsmtExposureModel, newdata = data.frame(BsmtExposure_NA[names(BsmtExposure_NA) != "BsmtExposure"]))
BsmtExposure_NA$BsmtExposure <- BsmtExposure_new
dataBsmtExposure <- rbind(BsmtExposure_nonNA, BsmtExposure_NA)

dataFull_nonNA <- dataBsmtExposure
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "BsmtExposure"]

#################################################
###"GarageType", factors --> randomForest
dataGarageType <- data.frame(dataFull_nonNA, GarageType = dataFull_NA$GarageType)
GarageType_nonNA <- droplevels(subset(dataGarageType, !is.na(dataGarageType$GarageType)))
GarageType_NA <- subset(dataGarageType, is.na(dataGarageType$GarageType))
GarageTypeModel <- randomForest(as.factor(GarageType) ~  ., data = GarageType_nonNA[(names(GarageType_nonNA) != "is_train") & (names(GarageType_nonNA) != "Id") & (names(GarageType_nonNA) != "Condition2") & (names(GarageType_nonNA) != "RoofMatl") & (names(GarageType_nonNA) != "MSSubClass")])

GarageType_new <- predict(GarageTypeModel, newdata = data.frame(GarageType_NA[names(GarageType_NA) != "GarageType"]))
GarageType_NA$GarageType <- GarageType_new
dataGarageType <- rbind(GarageType_nonNA, GarageType_NA)

dataFull_nonNA <- dataGarageType
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "GarageType"]

#################################################
###"GarageYrBlt", numeric --> linear regression
dataGarageYrBlt <- data.frame(dataFull_nonNA, GarageYrBlt = dataFull_NA$GarageYrBlt)
GarageYrBlt_nonNA <- subset(dataGarageYrBlt, !is.na(dataGarageYrBlt$GarageYrBlt))
GarageYrBlt_NA <- subset(dataGarageYrBlt, is.na(dataGarageYrBlt$GarageYrBlt))

fitGarageYrBlt <- lm(GarageYrBlt ~ . -1, data = GarageYrBlt_nonNA[(names(GarageYrBlt_nonNA) != "is_train") & (names(GarageYrBlt_nonNA) != "Id")])
my_anova <- anova(fitGarageYrBlt)
GarageYrBlt_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
GarageYrBltModel <- lm(GarageYrBlt ~ ., data = GarageYrBlt_nonNA[(names(GarageYrBlt_nonNA) != "is_train") & (names(GarageYrBlt_nonNA) != "Id") & ((names(GarageYrBlt_nonNA) == "GarageYrBlt") | (names(GarageYrBlt_nonNA) %in% GarageYrBlt_features))])

GarageYrBlt_new <- predict(GarageYrBltModel, newdata = data.frame(GarageYrBlt_NA[names(GarageYrBlt_NA) != "GarageYrBlt"]))
GarageYrBlt_NA$GarageYrBlt <- GarageYrBlt_new
dataGarageYrBlt <- rbind(GarageYrBlt_nonNA, GarageYrBlt_NA)

dataFull_nonNA <- dataGarageYrBlt
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "GarageYrBlt"]

#################################################
###"GarageFinish", factors --> randomForest
dataGarageFinish <- data.frame(dataFull_nonNA, GarageFinish = dataFull_NA$GarageFinish)
GarageFinish_nonNA <- droplevels(subset(dataGarageFinish, !is.na(dataGarageFinish$GarageFinish)))
GarageFinish_NA <- subset(dataGarageFinish, is.na(dataGarageFinish$GarageFinish))
GarageFinishModel <- randomForest(as.factor(GarageFinish) ~  ., data = GarageFinish_nonNA[(names(GarageFinish_nonNA) != "is_train") & (names(GarageFinish_nonNA) != "Id")& (names(GarageFinish_nonNA) != "Condition2") & (names(GarageFinish_nonNA) != "RoofMatl")])

GarageFinish_new <- predict(GarageFinishModel, newdata = data.frame(GarageFinish_NA[names(GarageFinish_NA) != "GarageFinish"]))
GarageFinish_NA$GarageFinish <- GarageFinish_new
dataGarageFinish <- rbind(GarageFinish_nonNA, GarageFinish_NA)

dataFull_nonNA <- dataGarageFinish
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "GarageFinish"]

#################################################
###"GarageQual", factors --> randomForest
dataGarageQual <- data.frame(dataFull_nonNA, GarageQual = dataFull_NA$GarageQual)
GarageQual_nonNA <- droplevels(subset(dataGarageQual, !is.na(dataGarageQual$GarageQual)))
GarageQual_NA <- subset(dataGarageQual, is.na(dataGarageQual$GarageQual))
GarageQualModel <- randomForest(as.factor(GarageQual) ~  ., data = GarageQual_nonNA[(names(GarageQual_nonNA) != "is_train") & (names(GarageQual_nonNA) != "Id") & (names(GarageQual_nonNA) != "Heating") & (names(GarageQual_nonNA) != "BedroomAbvGr") & (names(GarageQual_nonNA) != "TotRmsAbvGrd")])

GarageQual_new <- predict(GarageQualModel, newdata = data.frame(GarageQual_NA[names(GarageQual_NA) != "GarageQual"]))
GarageQual_NA$GarageQual <- GarageQual_new
dataGarageQual <- rbind(GarageQual_nonNA, GarageQual_NA)

dataFull_nonNA <- dataGarageQual
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "GarageQual"]

#################################################
###"GarageCond", factors --> randomForest
dataGarageCond <- data.frame(dataFull_nonNA, GarageCond = dataFull_NA$GarageCond)
GarageCond_nonNA <- droplevels(subset(dataGarageCond, !is.na(dataGarageCond$GarageCond)))
GarageCond_NA <- subset(dataGarageCond, is.na(dataGarageCond$GarageCond))
GarageCondModel <- randomForest(as.factor(GarageCond) ~  ., data = GarageCond_nonNA[(names(GarageCond_nonNA) != "is_train") & (names(GarageCond_nonNA) != "Id") & (names(GarageCond_nonNA) != "TotRmsAbvGrd")])

GarageCond_new <- predict(GarageCondModel, newdata = data.frame(GarageCond_NA[names(GarageCond_NA) != "GarageCond"]))
GarageCond_NA$GarageCond <- GarageCond_new
dataGarageCond <- rbind(GarageCond_nonNA, GarageCond_NA)

dataFull_nonNA <- dataGarageCond
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "GarageCond"]

#################################################
###"LotFrontage", numeric --> linear regression
dataLotFrontage <- data.frame(dataFull_nonNA, LotFrontage = dataFull_NA$LotFrontage)
LotFrontage_nonNA <- subset(dataLotFrontage, !is.na(dataLotFrontage$LotFrontage))
LotFrontage_NA <- subset(dataLotFrontage, is.na(dataLotFrontage$LotFrontage))

fitLotFrontage <- lm(LotFrontage ~ . -1, data = LotFrontage_nonNA[(names(LotFrontage_nonNA) != "is_train") & (names(LotFrontage_nonNA) != "Id") & (names(LotFrontage_nonNA) != "Utilities")])
my_anova <- anova(fitLotFrontage)
LotFrontage_features <- row.names(subset(my_anova, my_anova$`Pr(>F)` < 0.001))
LotFrontageModel <- lm(LotFrontage ~ ., data = LotFrontage_nonNA[(names(LotFrontage_nonNA) != "is_train") & (names(LotFrontage_nonNA) != "Id") & ((names(LotFrontage_nonNA) == "LotFrontage") | (names(LotFrontage_nonNA) %in% LotFrontage_features))])

LotFrontage_new <- predict(LotFrontageModel, newdata = data.frame(LotFrontage_NA[names(LotFrontage_NA) != "LotFrontage"]))
LotFrontage_NA$LotFrontage <- LotFrontage_new
dataLotFrontage <- rbind(LotFrontage_nonNA, LotFrontage_NA)

dataFull_nonNA <- dataLotFrontage
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "LotFrontage"]

#################################################
###"FireplaceQu", factors --> randomForest
dataFireplaceQu <- data.frame(dataFull_nonNA, FireplaceQu = dataFull_NA$FireplaceQu)
FireplaceQu_nonNA <- droplevels(subset(dataFireplaceQu, !is.na(dataFireplaceQu$FireplaceQu)))
FireplaceQu_NA <- subset(dataFireplaceQu, is.na(dataFireplaceQu$FireplaceQu))

sum(dataFull$Fireplaces==0)
sum(is.na(dataFull$FireplaceQu))

FireplaceQu_NA$FireplaceQu <- rep(as.factor("None"), length(FireplaceQu_NA$FireplaceQu))
dataFireplaceQu <- rbind(FireplaceQu_nonNA, FireplaceQu_NA)

dataFull_nonNA <- dataFireplaceQu
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "FireplaceQu"]

#################################################
###"Fence", factors --> randomForest
dataFence <- data.frame(dataFull_nonNA, Fence = dataFull_NA$Fence)
Fence_nonNA <- droplevels(subset(dataFence, !is.na(dataFence$Fence)))
Fence_NA <- subset(dataFence, is.na(dataFence$Fence))

Fence_NA$Fence <- rep(as.factor("NoFence"), length(Fence_NA$Fence))
dataFence <- rbind(Fence_nonNA, Fence_NA)

dataFull_nonNA <- dataFence
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "Fence"]

#################################################
###"Alley", factors --> randomForest
dataAlley <- data.frame(dataFull_nonNA, Alley = dataFull_NA$Alley)
Alley_nonNA <- droplevels(subset(dataAlley, !is.na(dataAlley$Alley)))
Alley_NA <- subset(dataAlley, is.na(dataAlley$Alley))

Alley_NA$Alley <- rep(as.factor("NoAlley"), length(Alley_NA$Alley))
dataAlley <- rbind(Alley_nonNA, Alley_NA)

dataFull_nonNA <- dataAlley
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "Alley"]

#################################################
###"MiscFeature", factors --> randomForest
dataMiscFeature <- data.frame(dataFull_nonNA, MiscFeature = dataFull_NA$MiscFeature)
MiscFeature_nonNA <- droplevels(subset(dataMiscFeature, !is.na(dataMiscFeature$MiscFeature)))
MiscFeature_NA <- subset(dataMiscFeature, is.na(dataMiscFeature$MiscFeature))

MiscFeature_NA$MiscFeature <- rep(as.factor("NoMiscFeature"), length(MiscFeature_NA$MiscFeature))
dataMiscFeature <- rbind(MiscFeature_nonNA, MiscFeature_NA)

dataFull_nonNA <- dataMiscFeature
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "MiscFeature"]

#################################################
###"PoolQC", factors --> randomForest
#################################################
dataPoolQC <- data.frame(dataFull_nonNA, PoolQC = dataFull_NA$PoolQC)
PoolQC_nonNA <- droplevels(subset(dataPoolQC, !is.na(dataPoolQC$PoolQC)))
PoolQC_NA_nopool <- subset(dataPoolQC, is.na(dataPoolQC$PoolQC) & (dataPoolQC$PoolArea==0))
PoolQC_NA_pool <- subset(dataPoolQC, is.na(dataPoolQC$PoolQC) & (dataPoolQC$PoolArea>0))

PoolQCModel <- randomForest(as.factor(PoolQC) ~  ., data = data.frame(PoolQC_nonNA[,sapply(PoolQC_nonNA, is.numeric) | sapply(PoolQC_nonNA, is.integer)], PoolQC = PoolQC_nonNA$PoolQC))

PoolQC_new <- predict(PoolQCModel, newdata = data.frame(PoolQC_NA_pool[,sapply(PoolQC_NA_pool, is.numeric) | sapply(PoolQC_NA_pool, is.integer)]))
PoolQC_NA_pool$PoolQC <- PoolQC_new

PoolQC_NA_nopool$PoolQC <- rep(as.factor("NoPool"), length(PoolQC_NA_nopool$PoolQC))

dataPoolQC <- rbind(PoolQC_nonNA, PoolQC_NA_pool, PoolQC_NA_nopool)

dataFull_nonNA <- dataPoolQC
dataFull_NA <- dataFull_NA[names(dataFull_NA) != "PoolQC"]

#####################################################################
# Write back to training and test data
dataFull <- dataFull_nonNA
dataFinal <- merge(dataFull, dataFullA[names(dataFullA) == "Id" | names(dataFullA) == "SalePrice"], by="Id")

train <- droplevels(subset(dataFinal, dataFinal$is_train == TRUE))
test <- subset(dataFinal, dataFinal$is_train == FALSE)

# Remove is_train column
train <- train[names(train) != "is_train"]
test <- test[names(test) != "is_train"]

#####################################################################
# Perserve the data
#####################################################################

outputTrain <- data.frame(train)
write.csv(outputTrain, file = "trainClean.csv", row.names = FALSE)

outputTest <- data.frame(test)
write.csv(outputTest, file = "testClean.csv", row.names = FALSE)

outputFinal <- data.frame(dataFinal)
write.csv(outputFinal, file = "traintestClean.csv", row.names = FALSE)
