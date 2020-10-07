import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("your data file name here")

le = preprocessing.LabelEncoder()

# all required columns except predictor/class column
x-col1 = le.fit_transform(list(data["x-col1"]))
x-col2 = le.fit_transform(list(data["x-col2"]))
x-col3 = le.fit_transform(list(data["x-col3"]))
x-col4 = le.fit_transform(list(data["x-col4"]))
x-col5 = le.fit_transform(list(data["x-col5"]))
x-col6 = le.fit_transform(list(data["x-col6"]))

#class column
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(x-col1, x-col2, x-col3, x-col4, x-col5, x-col6))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors= '''what ever number you want, odd prefered for better score''')

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["class", "attribute", "labels"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], '''what ever number you want, odd prefered for better score''', True)
    print("N: ", n)