import numpy as np
import pandas as pd
from pycaret.classification import setup, compare_models
from pycaret.classification import predict_model, save_model, load_model
from sklearn.preprocessing import StandardScaler
from pycaret.classification import tune_model
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import cluster
import sklearn
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor


pd.set_option('display.width', 600)
pd.set_option('display.max_columns', 20)

#importing train & test files
train_data = pd.read_csv("C:/Users/Jaroslav/Downloads/train.csv")
test_data = pd.read_csv("C:/Users/Jaroslav/Downloads/test.csv")
whole_data = [train_data, test_data]



def preprocess_data(data):
    data = data[['PassengerId','Pclass', 'Sex', 'Age',  'SibSp', 'Parch', 'Fare', 'Embarked']]
    data =  pd.get_dummies(data, columns=['Sex'])
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    labels = [1, 2, 3, 4, 5]
    data['Fare'] = pd.cut(data['Fare'], 5, labels=labels)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    data['Embarked'].fillna(data['Embarked'].median(), inplace=True)
    data['Nrelatives'] = data['SibSp'] + data['Parch']
    labels = [1,2,3,4,5]
    data['Age'] = pd.cut(data['Age'], 5, labels=labels)

    return data


train_features = preprocess_data(train_data)
train_targets = train_data['Survived']
print(train_features)
test_features = preprocess_data(test_data)


'''
X_train, X_test, y_train, y_test = train_test_split(train_features, train_targets, train_size=0.75,  random_state=42, shuffle=False)
scaled_lr_model = LogisticRegression(random_state=42)
scaled_lr_model.fit(X_train, y_train)

print(scaled_lr_model.score(X_train, y_train))
print(scaled_lr_model.score(X_test, y_test))
'''

scaler = StandardScaler()
scaled_features = scaler.fit(train_features[['Pclass', 'Sex_female',  'Sex_male', 'Age', 'SibSp', 'Parch','Nrelatives', 'Fare', 'Embarked',  ]])
X_train = pd.DataFrame(scaler.transform(train_features[['Pclass', 'Sex_female',  'Sex_male', 'Age', 'SibSp', 'Parch','Nrelatives','Fare', 'Embarked',  ]]), columns=train_features[['Pclass', 'Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch','Nrelatives', 'Fare', 'Embarked', ]].columns)


scaled_test_features = scaler.transform(test_features[['Pclass', 'Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch','Nrelatives', 'Fare', 'Embarked', ]])
X_test = pd.DataFrame(scaler.transform(test_features[['Pclass', 'Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch','Nrelatives', 'Fare', 'Embarked', ]]), columns=test_features[['Pclass', 'Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch','Nrelatives', 'Fare', 'Embarked', ]].columns)



def cluster(X):
    kmeans = sklearn.cluster.KMeans(n_clusters=5, random_state=0, n_init="auto")
    kmeans.fit(X)
    return ((kmeans.labels_+1)/5)

X=X_train[['Sex_female', 'Sex_male', 'Pclass', ]]
X_train['Cluster']=cluster(X)
X=X_train[['Fare', 'Nrelatives', 'Pclass']]
X_train['Cluster2']=cluster(X)





#PyCaret/AutoML
exp_clf = setup(X_train, target=train_targets )
best = compare_models(sort='Accuracy')
print(best)


X=X_test[['Sex_female', 'Sex_male', 'Pclass', ]]
X_test['Cluster']=cluster(X)
X=X_test[['Fare', 'Nrelatives', 'Pclass']]
X_test['Cluster2']=cluster(X)


print(X_test)

predict_df = predict_model(best, X_test)
print(predict_df)

result_df = predict_df.copy()
result_df['Survived'] = predict_df['prediction_label']
result_df['PassengerId'] = test_features['PassengerId']
result_df.to_csv('Titanic_PyCaret.csv', columns=('PassengerId','Survived'), index=False)
