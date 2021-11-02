import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

trainData = pd.read_csv (r'./KS_model_training_data.csv', sep = ',')
trainData = trainData.drop(columns=['backers_count', 'converted_pledged_amount', 'pledged', 'usd_pledged'])
trainData.isnull().sum()

trainData = trainData.dropna()
np.where(trainData.applymap(lambda x: x == ''))

trainData['created_at'] = pd.to_datetime(trainData['created_at'],unit='s')
trainData['created_month'] = trainData.created_at.apply(lambda x: x.month)
trainData['deadline'] = pd.to_datetime(trainData['deadline'],unit='s')

#derived features
trainData['created_year'] = trainData.created_at.apply(lambda x: x.year)
trainData['blurb_length'] = trainData['blurb'].str.len()
trainData['name_length'] = trainData['name'].str.len()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report


testData = pd.read_csv (r'./KS_test_data.csv', sep = ';')
np.where(testData.applymap(lambda x: x == ''))
testData = trainData.dropna()

testData['name_length'] = testData['name'].str.len()
testData['blurb_length'] = testData['blurb'].str.len()
testData['created_at'] = pd.to_datetime(trainData['created_at'],unit='s')
testData['created_hour'] = testData.created_at.apply(lambda x: x.hour)
testData['created_month'] = testData.created_at.apply(lambda x: x.month)
testData['created_year'] = testData.created_at.apply(lambda x: x.year)
testData['deadline'] = pd.to_datetime(trainData['deadline'],unit='s')
testData['launched_at'] = pd.to_datetime(trainData['launched_at'],unit='s')

categoriesToEncode = ['category', 'subcategory', 'currency', 'country']
trainDataHotEncoded = pd.get_dummies(trainData, prefix='category', columns=categoriesToEncode)
testDataHotEncoded = pd.get_dummies(testData, prefix='category', columns=categoriesToEncode)


features = ['name_length', 'created_year', 'blurb_length']
X = trainDataHotEncoded[features]
y = trainDataHotEncoded['funded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
accuracy = knn.score(X_test, y_test)

print(classification_report(y_test, knn.predict(X_test)))
print("accuracy = " + str(round(100 * accuracy)) + "%")
cross_val_score(knn, X_train, y_train, cv=5)

