# takes new datapoint and compares euclidian distance between that data point and other data points
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)  # or use: df.dropna(inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)  # test size = 20%

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 2, 1, 2, 2, 3, 2, 1]])
# if I want to enter more example_measures it has to be list of lists, for example:
# np.array([4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 2, 1, 2, 3, 3, 2, 1]) and increase .reshape() on .reshape(2,-1)
# OR use len() method
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)
