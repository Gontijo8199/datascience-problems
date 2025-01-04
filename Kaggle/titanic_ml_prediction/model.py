import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
a = pd.read_csv("train.csv")
c = pd.read_csv("test.csv")

'''

https://www.kaggle.com/competitions/titanic

https://www.kaggle.com/code/rgontijof/titanic-surviving-predictions

made with love, by Rafael Gontijo

'''

# regularizing data

imputer = SimpleImputer(strategy='median')
a['Age'] = imputer.fit_transform(a[['Age']])
a['Embarked'] = a['Embarked'].fillna(a['Embarked'].mode()[0])


c['Age'] = imputer.fit_transform(c[['Age']])
c['Fare'] = imputer.fit_transform(c[['Fare']])

label = LabelEncoder()

for columns in ['Sex']:
    a[columns] = label.fit_transform(a[columns])
    c[columns] = label.fit_transform(c[columns])
    # male-> 1, female -> 0

for columns in ['Embarked']:
    a[columns] = label.fit_transform(a[columns])
    c[columns] = label.fit_transform(c[columns])

    # C = Cherbourg, Q = Queenstown, S = Southampton
    # 0            , 1             , 2
    


# selecting whats going to be trained

b = ['Name', 'Ticket', 'Cabin'] # we can handle titles in name later.

a_updated = a.drop(columns=b)

# handling and tokenizing title

a_updated['Title'] = a['Name'].str.extract(r',\s*([^\.]*)\s*\.') # take tittles into consideration
titles = a_updated['Title'].unique()
title_tokenizer = {title: idx for idx, title in enumerate(titles)}
a_updated['Title'] = a_updated['Title'].map(title_tokenizer)


a_updated['Cabin'] = a['Cabin'].fillna('0')  # handle missing values
cabins = a_updated['Cabin'].unique()
cabin_tokenizer = {cabin: idx for idx, cabin in enumerate(cabins)}
a_updated['Cabin'] = a_updated['Cabin'].map(cabin_tokenizer)


a_y = a_updated['Survived']
a_X = a_updated.drop(columns='Survived', axis=1)


c_updated= c.drop(columns=b)

# handling and tokenizing title

c_updated['Title'] = c['Name'].str.extract(r',\s*([^\.]*)\s*\.') # take tittles into consideration
titles = c_updated['Title'].unique()
title_tokenizer = {title: idx for idx, title in enumerate(titles)}
c_updated['Title'] = c_updated['Title'].map(title_tokenizer)


c_updated['Cabin'] = c['Cabin'].fillna('0')  # handle missing values
cabins = c_updated['Cabin'].unique()
cabin_tokenizer = {cabin: idx for idx, cabin in enumerate(cabins)}
c_updated['Cabin'] = c_updated['Cabin'].map(cabin_tokenizer)

c_id = c_updated['PassengerId']




# modelling

X_train, X_val, y_train, y_val = train_test_split(a_X, a_y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# testing

c_predictions = random_forest.predict(c_updated)

submission = pd.DataFrame({
    'PassengerId': c_id,
    'Survived': c_predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created successfully!")



