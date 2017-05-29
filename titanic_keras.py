import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.metrics import categorical_accuracy as accuracy

features_train = pd.read_csv('train.csv')
features_test = pd.read_csv('test.csv')
labels_train = features_train['Survived']
features_train.drop('Survived', axis=1, inplace=True)

combined_features = features_train.append(features_test)
combined_features.reset_index(inplace=True)
combined_features.drop('index', axis=1, inplace=True)


#processing the importance- titles
combined_features['Title']= combined_features['Name'].map(
    lambda name: name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
combined_features['Title']=combined_features.Title.map(Title_Dictionary)


# Processing the ages
grouped_train = combined_features.head(891).groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()

grouped_test = combined_features.iloc[891:].groupby(['Sex','Pclass','Title'])
grouped_median_test = grouped_test.median()
def fillAges(row, grouped_median):
    if row['Sex'] == 'female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 1, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 1, 'Mrs']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['female', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['female', 1, 'Royalty']['Age']

    elif row['Sex'] == 'female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 2, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 2, 'Mrs']['Age']

    elif row['Sex'] == 'female' and row['Pclass'] == 3:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 3, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 3, 'Mrs']['Age']

    elif row['Sex'] == 'male' and row['Pclass'] == 1:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 1, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 1, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['male', 1, 'Royalty']['Age']

    elif row['Sex'] == 'male' and row['Pclass'] == 2:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 2, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 2, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 2, 'Officer']['Age']

    elif row['Sex'] == 'male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 3, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 3, 'Mr']['Age']

combined_features.head(891).Age = combined_features.head(891).apply(lambda r: fillAges(r, grouped_median_train) if np.isnan(r['Age'])
    else r['Age'], axis=1)

combined_features.iloc[891:].Age = combined_features.iloc[891:].apply(lambda r: fillAges(r, grouped_median_test) if np.isnan(r['Age'])
    else r['Age'], axis=1)



# Processing names: drop names and encode the titles using dummy encoding from pandas
combined_features.drop('Name', axis=1, inplace=True)
dummy_titles = pd.get_dummies(combined_features['Title'], prefix='Title')
combined_features = pd.concat([combined_features, dummy_titles], axis=1)
combined_features.drop('Title', axis=1, inplace=True)


# Processing Fares: fill empty spaces with mean
combined_features.head(891).Fare.fillna(combined_features.head(891).Fare.mean(), inplace=True)
combined_features.iloc[891:].Fare.fillna(combined_features.iloc[891:].Fare.mean(), inplace=True)


#Processing Embarked: drop it
combined_features.drop('Embarked', axis=1, inplace=True)


#Processing Cabin:
combined_features.Cabin.fillna('U', inplace=True)

# mapping each Cabin value with the cabin letter
combined_features['Cabin'] = combined_features['Cabin'].map(lambda c: c[0])

cabin_dummies = pd.get_dummies(combined_features['Cabin'], prefix='Cabin')
combined_features = pd.concat([combined_features, cabin_dummies], axis=1)
combined_features.drop('Cabin', axis=1, inplace=True)


# Processing sex
combined_features['Sex'] = combined_features['Sex'].map({'male':1,'female':0})


# Processing pclass
pclass_dummies = pd.get_dummies(combined_features['Pclass'], prefix="Pclass")
combined_features = pd.concat([combined_features, pclass_dummies], axis=1)
combined_features.drop('Pclass', axis=1, inplace=True)

# Processing Ticket
combined_features.drop('Ticket', axis=1, inplace=True)

# Processing family
combined_features['FamilySize'] = combined_features['Parch'] + combined_features['SibSp'] + 1
combined_features['Singleton'] = combined_features['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combined_features['SmallFamily'] = combined_features['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
combined_features['LargeFamily'] = combined_features['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
combined_features.drop('Parch', axis=1, inplace=True)
combined_features.drop('SibSp', axis=1, inplace=True)
combined_features.drop('PassengerId', axis=1, inplace=True)

#scaling
scale=max(combined_features['Age'])
combined_features['Age']/=scale
mean=np.mean(combined_features['Age'])
combined_features['Age']-=mean

scale=max(combined_features['Fare'])
combined_features['Fare']/=scale
mean=np.mean(combined_features['Fare'])
combined_features['Fare']-=mean

#split training and testing data
features_train=combined_features[:891]
features_test=combined_features[891:]


input_dim = features_train.shape[1]
#mlp
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
# for i in range(100):
#     model.add(Dense(128))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.15))
model.add(Dense(2))
model.add(Activation('softmax'))
labels_train = np_utils.to_categorical(labels_train)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

features_train= np.array(features_train)
features_test= np.array(features_test)
print("Training...")
model.fit(features_train, labels_train, epochs=100, validation_split=0.1)

print("Generating test predictions...")
preds = model.predict_classes(features_test, verbose=0)


def write_preds(preds, fname):
    pd.DataFrame({"PassengerID": list(range(892,891+418+1)), "Survived": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "result-1.csv")