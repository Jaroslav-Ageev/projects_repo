import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from pycaret.regression import setup, compare_models
from pycaret.regression import predict_model, save_model, load_model
from sklearn.preprocessing import StandardScaler
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



print(train_data['Foundation'].unique().shape)
print(train_data['Heating'].unique().shape)
print(train_data['Neighborhood'].unique().shape)

#one_hot_encoded_training_predictors = pd.get_dummies(train_features.select_dtypes(include='object'))
#print(one_hot_encoded_training_predictors)
#print(one_hot_encoded_training_predictors.shape)

def preprocess_data(data):

    data = data[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',  'LotShape', 'LandContour',
                 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
                 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
                 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt',
                 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',   'MiscVal', 'MoSold', 'YrSold',
                 'SaleType', 'SaleCondition','SalePrice']]

    data['Foundation'] = data['Foundation'].map({'PConc': 0, 'CBlock': 1, 'BrkTil': 2, 'Stone': 3, 'Slab': 4, 'Wood': 5})
    data['Foundation'] = data['Foundation'].astype(int)

    data['Heating'] = data['Heating'].map({'GasA': 0, 'GasW': 1, 'Wall': 2, 'OthW': 3, 'Grav': 4, 'Floor': 5})
    data['Heating'] = data['Heating'].astype(int)


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
    data['Exterior2nd'] = OrdinalEncoder().fit_transform(data['Exterior2nd'].values.reshape(-1, 1))
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
    data['GarageYrBlt'].fillna(data['GarageYrBlt'].mode(), inplace=True)
    data['GarageCond'].fillna(data['GarageCond'].mode(), inplace=True)
    data['GarageType'].fillna(data['GarageType'].mode(), inplace=True)
    data['GarageFinish'].fillna(data['GarageFinish'].mode(), inplace=True)
    data['GarageQual'].fillna(data['GarageQual'].mode(), inplace=True)
    data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode(), inplace=True)
    data['BsmtExposure'].fillna(data['BsmtExposure'].mean(), inplace=True)
    data['BsmtQual'].fillna(data['BsmtQual'].mode(), inplace=True)
    data['BsmtCond'].fillna(data['BsmtCond'].mode(), inplace=True)
    data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode(), inplace=True)
    data['MasVnrArea'].fillna(data['MasVnrArea'].median(), inplace=True)
    data['MasVnrType'].fillna(data['MasVnrType'].mode(), inplace=True)
    data['Electrical'].fillna(data['MasVnrType'].mode(), inplace=True)

    return data

def preprocess_data_test(data):

    data = data[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',  'LotShape', 'LandContour',
                 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
                 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
                 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt',
                 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',   'MiscVal', 'MoSold', 'YrSold',
                 'SaleType', 'SaleCondition']]

    data['Foundation'] = data['Foundation'].map({'PConc': 0, 'CBlock': 1, 'BrkTil': 2, 'Stone': 3, 'Slab': 4,'Wood': 5})
    data['Foundation'] = data['Foundation'].astype(int)

    data['Heating'] = data['Heating'].map({'GasA': 0, 'GasW': 1, 'Wall': 2, 'OthW': 3, 'Grav': 4, 'Floor': 5})
    data['Heating'] = data['Heating'].astype(int)

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
    data['Exterior2nd'] = OrdinalEncoder().fit_transform(data['Exterior2nd'].values.reshape(-1, 1))
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
    data['GarageYrBlt'].fillna(data['GarageYrBlt'].mode(), inplace=True)
    data['GarageCond'].fillna(data['GarageCond'].mode(), inplace=True)
    data['GarageType'].fillna(data['GarageType'].mode(), inplace=True)
    data['GarageFinish'].fillna(data['GarageFinish'].mode(), inplace=True)
    data['GarageQual'].fillna(data['GarageQual'].mode(), inplace=True)
    data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode(), inplace=True)
    data['BsmtExposure'].fillna(data['BsmtExposure'].mean(), inplace=True)
    data['BsmtQual'].fillna(data['BsmtQual'].mode(), inplace=True)
    data['BsmtCond'].fillna(data['BsmtCond'].mode(), inplace=True)
    data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode(), inplace=True)
    data['MasVnrArea'].fillna(data['MasVnrArea'].median(), inplace=True)
    data['MasVnrType'].fillna(data['MasVnrType'].mode(), inplace=True)
    data['Electrical'].fillna(data['MasVnrType'].mode(), inplace=True)

    return data

td = preprocess_data(train_features)
td_test = preprocess_data_test(test_features)

# new features

td['SqFtPerRoom'] = td["GrLivArea"] / (td["TotRmsAbvGrd"] +
                                               td["FullBath"] +
                                               td["HalfBath"] +
                                               td["KitchenAbvGr"])
td['Total_Home_Quality'] = td['OverallQual'] + td['OverallCond']
td["Age"] = td["YrSold"] - td["YearBuilt"]

td_test['SqFtPerRoom'] = td_test["GrLivArea"] / (td_test["TotRmsAbvGrd"] +
                                               td_test["FullBath"] +
                                               td_test["HalfBath"] +
                                               td_test["KitchenAbvGr"])
td_test['Total_Home_Quality'] = td_test['OverallQual'] + td_test['OverallCond']
td_test["Age"] = td_test["YrSold"] - td_test["YearBuilt"]

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

predict_td = predict_model(best, X_test)
print(predict_td)



#test
result_df = predict_td.copy()
result_df['SalePrice'] = result_df['prediction_label']
result_df['Id'] = td_test['Id']
result_df.to_csv('house_prices_results_7.csv', columns=('Id','SalePrice'), index=False)

