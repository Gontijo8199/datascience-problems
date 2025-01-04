import pandas as pd
from sklearn.metrics import r2_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# import matplotlib.pyplot as plt
# import seaborn as sns

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Regularizing data

# dropping columns with too many missing values
data_train = train.drop(columns = ['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature'])
data_test = test.drop(columns = ['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature']) 


data_train_numeric = data_train.select_dtypes(include=['int64','float64']).columns
data_train_categorical = data_train.select_dtypes(include=['object', 'O']).columns

data_test_numeric = data_test.select_dtypes(include=['int64','float64']).columns
data_test_categorical = data_test.select_dtypes(include=['object', 'O']).columns

# fill numeric columns with median

inputer = SimpleImputer(strategy='median')
data_train[data_train_numeric] = inputer.fit_transform(data_train[data_train_numeric])

data_test[data_test_numeric] = inputer.fit_transform(data_test[data_test_numeric])

# fill categorical columns with mode

columns_to_fill_train = data_train_categorical
columns_to_fill_test = data_test_categorical

for column in columns_to_fill_train:
    data_train[column] = data_train[column].fillna(data_train[column].mode()[0])

for column in columns_to_fill_test:
    data_test[column] = data_test[column].fillna(data_test[column].mode()[0])

# a = data_train['SalePrice']
# b = data_train.drop(columns=['SalePrice'])
# b = b[data_train_numeric.drop('SalePrice')]

# sale price distribution    
# bar = sns.distplot(data_train['SalePrice'])
# plt.show()

#correlation sale price with other features

#b.corrwith(a).plot.bar(figsize=(16,9), title='correlation with price', fontsize=10, rot=45)
#plt.show()

#heatmap = sns.heatmap(data=b.corr(), cmap='YlGnBu', annot= False, linewidths=1)
#plt.show()

# handling categorical columns

data_train = pd.get_dummies(data_train, columns=data_train_categorical, drop_first=True)
data_test = pd.get_dummies(data_test, columns=data_test_categorical, drop_first=True)

# Align the train and test dataframes by the columns
data_train, data_test = data_train.align(data_test, join='left', axis=1, fill_value=0)

y = data_train['SalePrice']
X = data_train.drop(columns=['SalePrice'])


scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




regressor_xgb = XGBRegressor()
regressor_xgb.fit(X_train, y_train)

y_pred = regressor_xgb.predict(X_test)

y_pred_2 = regressor_xgb.predict(data_test.drop(columns=['SalePrice']))
score = r2_score(y_test, y_pred)

data_test['Id'] = data_test['Id'].astype(int)
# why tf did it not format this as an integer in the first place?

submission = pd.DataFrame({
    'Id': data_test['Id'],
    'SalePrice': y_pred_2
})

submission.to_csv("submission.csv", index=False)

print(score)

