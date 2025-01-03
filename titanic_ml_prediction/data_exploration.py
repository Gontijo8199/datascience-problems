import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

a = pd.read_csv("train.csv")
c = pd.read_csv("test.csv")



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

a_y = a_updated['Survived']
a_X = a_updated.drop(columns='Survived', axis=1)


c_updated= c.drop(columns=b)

# handling and tokenizing title

c_updated['Title'] = c['Name'].str.extract(r',\s*([^\.]*)\s*\.') # take tittles into consideration
titles = c_updated['Title'].unique()
title_tokenizer = {title: idx for idx, title in enumerate(titles)}
c_updated['Title'] = c_updated['Title'].map(title_tokenizer)

c_id = c_updated['PassengerId']


