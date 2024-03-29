import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from pycaret.regression import setup, compare_models
from pycaret.regression import predict_model, save_model, load_model
from pycaret.regression import tune_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns

#importing train & test files
train_data = pd.read_csv("C:/Users/Jaroslav/anaconda3/exercises/pycaret_venv/pc_env/House_prices/train.csv")
test_data = pd.read_csv("C:/Users/Jaroslav/anaconda3/exercises/pycaret_venv/pc_env/House_prices/test.csv")


pd.set_option('display.width', 4000)
pd.set_option('display.max_columns', 100)
#pd.set_option('display.max_rows', 1000)

print(train_data.shape)
print(test_data.shape)
print(train_data.columns)
print(train_data.info())
print(train_data.describe())

list_null = train_data.isna().sum().sort_values(ascending=False)[:20]
print(list_null)


train_features = train_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', ], axis=1)
train_target = train_data['SalePrice']



test_features = test_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', ], axis=1)




def preprocess_data(data):

    data = data[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',  'LotShape', 'LandContour',
                 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
                 'Exterior1st', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '2ndFlrSF',
                 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                 'KitchenAbvGr', 'KitchenQual', 'Functional', 'Fireplaces', 'GarageType',
                 'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',   'MiscVal', 'MoSold', 'YrSold',
                 'SaleType', 'SaleCondition', 'TotRmsAbvGrd', 'SalePrice']]


    data['Foundation'] = OrdinalEncoder().fit_transform(data['Foundation'].values.reshape(-1, 1))
    data['Heating'] = OrdinalEncoder().fit_transform(data['Heating'].values.reshape(-1, 1))
    data['BsmtQual'] = OrdinalEncoder().fit_transform(data['BsmtQual'].values.reshape(-1, 1))
    data['MSZoning'] = OrdinalEncoder().fit_transform(data['MSZoning'].values.reshape(-1, 1))
    data['Street'] = OrdinalEncoder().fit_transform(data['Street'].values.reshape(-1, 1))
    data['LotShape'] = OrdinalEncoder().fit_transform(data['LotShape'].values.reshape(-1, 1))
    data['LandContour'] = OrdinalEncoder().fit_transform(data['LandContour'].values.reshape(-1, 1))
    data['Utilities'] = OrdinalEncoder().fit_transform(data['Utilities'].values.reshape(-1, 1))
    data['LotConfig'] = OrdinalEncoder().fit_transform(data['LotConfig'].values.reshape(-1, 1))
    data['LandSlope'] = OrdinalEncoder().fit_transform(data['LandSlope'].values.reshape(-1, 1))
    data['Neighborhood'] = OrdinalEncoder().fit_transform(data['Neighborhood'].values.reshape(-1, 1))
    data['Condition1'] = OrdinalEncoder().fit_transform(data['Condition1'].values.reshape(-1, 1))
    data['Condition2'] = OrdinalEncoder().fit_transform(data['Condition2'].values.reshape(-1, 1))
    data['BldgType'] = OrdinalEncoder().fit_transform(data['BldgType'].values.reshape(-1, 1))
    data['HouseStyle'] = OrdinalEncoder().fit_transform(data['HouseStyle'].values.reshape(-1, 1))
    data['RoofStyle'] = OrdinalEncoder().fit_transform(data['RoofStyle'].values.reshape(-1, 1))
    data['RoofMatl'] = OrdinalEncoder().fit_transform(data['RoofMatl'].values.reshape(-1, 1))
    data['Exterior1st'] = OrdinalEncoder().fit_transform(data['Exterior1st'].values.reshape(-1, 1))
    data['MasVnrType'] = OrdinalEncoder().fit_transform(data['MasVnrType'].values.reshape(-1, 1))
    data['ExterQual'] = OrdinalEncoder().fit_transform(data['ExterQual'].values.reshape(-1, 1))
    data['ExterCond'] = OrdinalEncoder().fit_transform(data['ExterCond'].values.reshape(-1, 1))
    data['BsmtCond'] = OrdinalEncoder().fit_transform(data['BsmtCond'].values.reshape(-1, 1))
    data['BsmtExposure'] = OrdinalEncoder().fit_transform(data['BsmtExposure'].values.reshape(-1, 1))
    data['BsmtFinType1'] = OrdinalEncoder().fit_transform(data['BsmtFinType1'].values.reshape(-1, 1))
    data['BsmtFinType2'] = OrdinalEncoder().fit_transform(data['BsmtFinType2'].values.reshape(-1, 1))
    data['HeatingQC'] = OrdinalEncoder().fit_transform(data['HeatingQC'].values.reshape(-1, 1))
    data['CentralAir'] = OrdinalEncoder().fit_transform(data['CentralAir'].values.reshape(-1, 1))
    data['Electrical'] = OrdinalEncoder().fit_transform(data['Electrical'].values.reshape(-1, 1))
    data['KitchenQual'] = OrdinalEncoder().fit_transform(data['KitchenQual'].values.reshape(-1, 1))
    data['Functional'] = OrdinalEncoder().fit_transform(data['Functional'].values.reshape(-1, 1))
    data['GarageType'] = OrdinalEncoder().fit_transform(data['GarageType'].values.reshape(-1, 1))
    data['GarageFinish'] = OrdinalEncoder().fit_transform(data['GarageFinish'].values.reshape(-1, 1))
    data['GarageQual'] = OrdinalEncoder().fit_transform(data['GarageQual'].values.reshape(-1, 1))
    data['GarageCond'] = OrdinalEncoder().fit_transform(data['GarageCond'].values.reshape(-1, 1))
    data['PavedDrive'] = OrdinalEncoder().fit_transform(data['PavedDrive'].values.reshape(-1, 1))
    data['SaleType'] = OrdinalEncoder().fit_transform(data['SaleType'].values.reshape(-1, 1))
    data['SaleCondition'] = OrdinalEncoder().fit_transform(data['SaleCondition'].values.reshape(-1, 1))

    # filling in missing entries -NaN
    data['LotFrontage'].fillna(data['LotFrontage'].mean(), inplace=True)
    data['GarageCond'].fillna(data['GarageCond'].mode()[0], inplace=True)
    data['GarageType'].fillna(data['GarageType'].mode()[0], inplace=True)
    data['GarageFinish'].fillna(data['GarageFinish'].mode()[0], inplace=True)
    data['GarageQual'].fillna(data['GarageQual'].mode()[0], inplace=True)
    data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0], inplace=True)
    data['BsmtExposure'].fillna(data['BsmtExposure'].mean(), inplace=True)
    data['BsmtQual'].fillna(data['BsmtQual'].mode()[0], inplace=True)
    data['BsmtCond'].fillna(data['BsmtCond'].mode()[0], inplace=True)
    data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode()[0], inplace=True)
    data['MasVnrArea'].fillna(data['MasVnrArea'].median(), inplace=True)
    data['MasVnrType'].fillna(data['MasVnrType'].mode()[0], inplace=True)
    data['Electrical'].fillna(data['Electrical'].mode()[0], inplace=True)

    return data

def preprocess_data_test(data):
    data = data[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour',
                 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
                 'Exterior1st', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '2ndFlrSF',
                 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                 'KitchenAbvGr', 'KitchenQual', 'Functional', 'Fireplaces', 'GarageType',
                 'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold',
                 'SaleType', 'SaleCondition', 'TotRmsAbvGrd',]]

    data['Foundation'] = OrdinalEncoder().fit_transform(data['Foundation'].values.reshape(-1, 1))
    data['Heating'] = OrdinalEncoder().fit_transform(data['Heating'].values.reshape(-1, 1))
    data['BsmtQual'] = OrdinalEncoder().fit_transform(data['BsmtQual'].values.reshape(-1, 1))
    data['MSZoning'] = OrdinalEncoder().fit_transform(data['MSZoning'].values.reshape(-1, 1))
    data['Street'] = OrdinalEncoder().fit_transform(data['Street'].values.reshape(-1, 1))
    data['LotShape'] = OrdinalEncoder().fit_transform(data['LotShape'].values.reshape(-1, 1))
    data['LandContour'] = OrdinalEncoder().fit_transform(data['LandContour'].values.reshape(-1, 1))
    data['Utilities'] = OrdinalEncoder().fit_transform(data['Utilities'].values.reshape(-1, 1))
    data['LotConfig'] = OrdinalEncoder().fit_transform(data['LotConfig'].values.reshape(-1, 1))
    data['LandSlope'] = OrdinalEncoder().fit_transform(data['LandSlope'].values.reshape(-1, 1))
    data['Neighborhood'] = OrdinalEncoder().fit_transform(data['Neighborhood'].values.reshape(-1, 1))
    data['Condition1'] = OrdinalEncoder().fit_transform(data['Condition1'].values.reshape(-1, 1))
    data['Condition2'] = OrdinalEncoder().fit_transform(data['Condition2'].values.reshape(-1, 1))
    data['BldgType'] = OrdinalEncoder().fit_transform(data['BldgType'].values.reshape(-1, 1))
    data['HouseStyle'] = OrdinalEncoder().fit_transform(data['HouseStyle'].values.reshape(-1, 1))
    data['RoofStyle'] = OrdinalEncoder().fit_transform(data['RoofStyle'].values.reshape(-1, 1))
    data['RoofMatl'] = OrdinalEncoder().fit_transform(data['RoofMatl'].values.reshape(-1, 1))
    data['Exterior1st'] = OrdinalEncoder().fit_transform(data['Exterior1st'].values.reshape(-1, 1))
    data['MasVnrType'] = OrdinalEncoder().fit_transform(data['MasVnrType'].values.reshape(-1, 1))
    data['ExterQual'] = OrdinalEncoder().fit_transform(data['ExterQual'].values.reshape(-1, 1))
    data['ExterCond'] = OrdinalEncoder().fit_transform(data['ExterCond'].values.reshape(-1, 1))
    data['BsmtCond'] = OrdinalEncoder().fit_transform(data['BsmtCond'].values.reshape(-1, 1))
    data['BsmtExposure'] = OrdinalEncoder().fit_transform(data['BsmtExposure'].values.reshape(-1, 1))
    data['BsmtFinType1'] = OrdinalEncoder().fit_transform(data['BsmtFinType1'].values.reshape(-1, 1))
    data['BsmtFinType2'] = OrdinalEncoder().fit_transform(data['BsmtFinType2'].values.reshape(-1, 1))
    data['HeatingQC'] = OrdinalEncoder().fit_transform(data['HeatingQC'].values.reshape(-1, 1))
    data['CentralAir'] = OrdinalEncoder().fit_transform(data['CentralAir'].values.reshape(-1, 1))
    data['Electrical'] = OrdinalEncoder().fit_transform(data['Electrical'].values.reshape(-1, 1))
    data['KitchenQual'] = OrdinalEncoder().fit_transform(data['KitchenQual'].values.reshape(-1, 1))
    data['Functional'] = OrdinalEncoder().fit_transform(data['Functional'].values.reshape(-1, 1))
    data['GarageType'] = OrdinalEncoder().fit_transform(data['GarageType'].values.reshape(-1, 1))
    data['GarageFinish'] = OrdinalEncoder().fit_transform(data['GarageFinish'].values.reshape(-1, 1))
    data['GarageQual'] = OrdinalEncoder().fit_transform(data['GarageQual'].values.reshape(-1, 1))
    data['GarageCond'] = OrdinalEncoder().fit_transform(data['GarageCond'].values.reshape(-1, 1))
    data['PavedDrive'] = OrdinalEncoder().fit_transform(data['PavedDrive'].values.reshape(-1, 1))
    data['SaleType'] = OrdinalEncoder().fit_transform(data['SaleType'].values.reshape(-1, 1))
    data['SaleCondition'] = OrdinalEncoder().fit_transform(data['SaleCondition'].values.reshape(-1, 1))

    # filling in missing entries -NaN
    data['LotFrontage'].fillna(data['LotFrontage'].mean(), inplace=True)
    data['GarageCond'].fillna(data['GarageCond'].mode()[0], inplace=True)
    data['GarageType'].fillna(data['GarageType'].mode()[0], inplace=True)
    data['GarageFinish'].fillna(data['GarageFinish'].mode()[0], inplace=True)
    data['GarageQual'].fillna(data['GarageQual'].mode()[0], inplace=True)
    data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0], inplace=True)
    data['BsmtExposure'].fillna(data['BsmtExposure'].mean(), inplace=True)
    data['BsmtQual'].fillna(data['BsmtQual'].mode()[0], inplace=True)
    data['BsmtCond'].fillna(data['BsmtCond'].mode()[0], inplace=True)
    data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode()[0], inplace=True)
    data['MasVnrArea'].fillna(data['MasVnrArea'].median(), inplace=True)
    data['MasVnrType'].fillna(data['MasVnrType'].mode()[0], inplace=True)
    data['Electrical'].fillna(data['MasVnrType'].mode()[0], inplace=True)

    data['Functional'].fillna(data['Functional'].mode()[0], inplace=True)
    data['Electrical'].fillna(("SBrkr"),inplace=True)
    data['KitchenQual'].fillna(data['KitchenQual'].mode()[0], inplace=True)
    data['Exterior1st'].fillna(data['Exterior1st'].mode()[0], inplace=True)
    data['SaleType'].fillna(data['SaleType'].mode()[0], inplace=True)
    data['MSZoning'].fillna(data['MSZoning'].mode()[0], inplace=True)
    data['Utilities'].fillna(data['Utilities'].mode()[0], inplace=True)
    data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].mode()[0], inplace=True)
    data['BsmtFullBath'].fillna(data['BsmtFullBath'].mode()[0], inplace=True)
    data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mean(), inplace=True)
    data['GarageArea'].fillna(data['GarageArea'].mean(), inplace=True)
    data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].mean(), inplace=True)
    data['Exterior1st'].fillna(data['Exterior1st'].mode()[0], inplace=True)
    data['BsmtFinSF2'].fillna(data['BsmtFinSF2'].mean(), inplace=True)
    data['SaleType'].fillna(data['SaleType'].mode()[0], inplace=True)
    data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mean(), inplace=True)



    return data

td = preprocess_data(train_features)
td_test = preprocess_data_test(test_features)

print(td_test.isna().sum().sort_values(ascending=False)[:20])

# new features td

td['SqFtPerRoom'] = td["GrLivArea"] / (td["TotRmsAbvGrd"] +
                                               td["FullBath"] +
                                               td["HalfBath"] +
                                               td["KitchenAbvGr"])
td['Total_Home_Quality'] = td['OverallQual'] + td['OverallCond']
td['GarageState'] = td['GarageQual'] + td['GarageCond']


# new features td_test
td_test['SqFtPerRoom'] = td_test["GrLivArea"] / (td_test["TotRmsAbvGrd"] +
                                               td_test["FullBath"] +
                                               td_test["HalfBath"] +
                                               td_test["KitchenAbvGr"])
td_test['Total_Home_Quality'] = td_test['OverallQual'] + td_test['OverallCond']
td_test['GarageState'] = td_test['GarageQual'] + td_test['GarageCond']




correlations = td.corr().abs().sort_values('SalePrice', ascending=False)[:20]
print(correlations)
print("TRAIN DATA")
print(td)


"""
dfCorr = td.corr()
filteredDf = dfCorr[((dfCorr >= .6) | (dfCorr <= -.6)) & (dfCorr !=1.000)]
plt.figure(figsize=(40,15))
sns.heatmap(filteredDf, annot=True, cmap="Reds")
plt.show()
"""

scaler = StandardScaler()
scaled_features = scaler.fit(td.drop(['SalePrice'], axis=1))
X_train = pd.DataFrame(scaler.transform(td.drop(['SalePrice'], axis=1)))

scaled_test_features = scaler.transform(td_test)
X_test = pd.DataFrame(scaler.transform(td_test))



#PyCaret/AutoML
exp_clf = setup(X_train, target=train_target)
best = compare_models(sort='RMSE')
print(best)



catboost = CatBoostRegressor()
gbr = GradientBoostingRegressor()
et = ExtraTreesRegressor()
lightgbm = LGBMRegressor()
rf = RandomForestRegressor()

catboost.fit(X_train, train_target)
a = catboost.predict(X_test)

gbr.fit(X_train, train_target)
b = gbr.predict(X_test)

et.fit(X_train, train_target)
c = et.predict(X_test)

lightgbm.fit(X=X_train, y=train_target)
d = lightgbm.predict(X_test)

rf.fit(X_train, train_target)
e = rf.predict(X_test)



predict_td = 0.4*a +0.15*b +0.05*c + 0.35*d + 0.05*e

"""
predict_td = predict_model(best, X_test)
"""
print(predict_td)
print(predict_td.shape)



#test
result_df = predict_td.copy()
result_df = pd.DataFrame(result_df, columns=['SalePrice'])
#result_df['SalePrice'] = result_df['prediction_label']
result_df['Id'] = td_test['Id']
result_df.to_csv('house_prices_results_6.csv', columns=('Id','SalePrice'), index=False)

