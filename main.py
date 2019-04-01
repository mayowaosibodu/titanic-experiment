
import pandas as pd
import keras

#Preprocessing.
data = pd.read_csv('titanic-experiment/train.csv')

target = data.Survived.to_numpy()


def create_dummies(df, column):
    dummies = pd.get_dummies(df[column], prefix= column)
    df = pd.concat([df, dummies], axis=1)
    return df

def preprocess(df):
    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    df = create_dummies(df, 'Sex')
    df = create_dummies(df, 'Pclass')
    df = create_dummies(df, 'Embarked')
    df = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    return df

def normalize(df, mode, mean, std):
    to_normalize = ['Age', 'SibSp', 'Parch', 'Fare']
    means = []
    stds = []
    for column in to_normalize:
        if mode== 'train':
            means.append(df[column].mean())
            stds.append(df[column].std())
            df[column] = (df[column]-df[column].mean())/df[column].std()
        else:
            index = to_normalize.index(column)
            df[column] = (df[column]-mean[index])/std[index]
    return df, means, stds


data = preprocess(data)
data = data.fillna(data.median())
data, means, stds = normalize(data, 'train', 0, 0)
data.to_csv('titanic-experiment/processed_train.csv')
data = data.to_numpy()


#Machine Learning.
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense
model.add(Dense(units=5, activation='relu', input_dim=data.shape[1]))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=1, activation='relu'))
# model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='mse',optimizer='sgd')
# model.compile(loss='binary_crossentropy',optimizer='sgd')
model.fit(data, target, epochs= 10000)



def pre_binarize(results):
    for i in range(len(results)):
        results[i] = list(results[i])
        results[i] = binarize(results[i][0])
    return results

def binarize(value) -> int:
    #print('\n Value passed to Binarize is:', value)
    if value < 0.5:
        return 0
    else:
        return 1

def evaluate(array1, array2):
    success = 0
    for i in range(len(array1)):
        # print(array1[i], array2[i])
        if array1[i] == array2[i]:
            success += 1
    return success/len(array1)


test_data = pd.read_csv('titanic-experiment/test.csv')


test_data = preprocess(test_data)
test_data, _, _ = normalize(test_data, 'test', means, stds)
# print(test_data[:5])

training_results = list(model.predict(data))
training_results = pre_binarize(training_results)
# print(training_results[:20])

test_results = list(model.predict(test_data))
test_results = pre_binarize(test_results)
test_results = pd.Series(test_results)
print(test_results[:20])
test_results.to_csv('titanic-experiment/predicted.csv')

print('Training Score:', evaluate(training_results,target))
