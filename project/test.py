import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

# pd.set_option('display.max_rows', None)
trainData = pd.read_csv (r'./KS_model_training_data.csv', sep = ',')
trainData = trainData.drop(columns=['backers_count', 'converted_pledged_amount', 'pledged', 'usd_pledged'])

trainData.isnull().sum()

trainData = trainData.dropna()
np.where(trainData.applymap(lambda x: x == ''))

trainData = trainData.head(100000)

trainData['created_at'] = pd.to_datetime(trainData['created_at'],unit='s')
trainData['created_month'] = trainData.created_at.apply(lambda x: x.month)
trainData['deadline'] = pd.to_datetime(trainData['deadline'],unit='s')

#derived features
trainData['created_year'] = trainData.created_at.apply(lambda x: x.year)
trainData['blurb_length'] = trainData['blurb'].str.len()
trainData['name_length'] = trainData['name'].str.len()

trainData.drop(['project_id', 'blurb', 'blurb_length','created_at', 'deadline', 'fx_rate', 'goal', 'launched_at', 'name', 'staff_pick', 
                'location', 'project_url', 'reward_url', 'created_month', 'name_length'], 1, inplace=True)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

testData = pd.read_csv (r'./KS_test_data.csv', sep = ';')
np.where(testData.applymap(lambda x: x == ''))
testData = testData.dropna()

testData['name_length'] = testData['name'].str.len()
testData['blurb_length'] = testData['blurb'].str.len()
testData['created_at'] = pd.to_datetime(testData['created_at'],unit='s')
testData['created_month'] = testData.created_at.apply(lambda x: x.month)
testData['created_year'] = testData.created_at.apply(lambda x: x.year)

testData.drop(['project_id', 'blurb','blurb_length','created_at', 'deadline', 'fx_rate', 'goal', 'launched_at', 'name', 'staff_pick', 
                'location', 'project_url', 'reward_url', 'created_month', 'name_length'], 1, inplace=True)

categoriesToEncode = ['category', 'subcategory', 'currency', 'country']
trainDataHotEncoded = pd.get_dummies(trainData, prefix='category', columns=categoriesToEncode)
testDataHotEncoded = pd.get_dummies(testData, prefix='category', columns=categoriesToEncode)


y = trainDataHotEncoded['funded']
X = trainDataHotEncoded.drop('funded', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
knn = KNeighborsClassifier(n_neighbors = 27)
cross_val_score(knn, X, y, cv=10)

# knn.fit(X_train,y_train)
# accuracy = knn.score(X_test, y_test)

# print(classification_report(y_test, knn.predict(X)))
# print("accuracy = " + str(round(100 * accuracy)) + "%")

print("starting")

# from sklearn.model_selection import GridSearchCV


#source
#https://medium.datadriveninvestor.com/k-nearest-neighbors-in-python-hyperparameters-tuning-716734bc557f

#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn_2 = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=10)
#Fit the model
best_model = clf.fit(X,y)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])