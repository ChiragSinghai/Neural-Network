import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names=['sepal-length','sepal-width','petal-length','petal-width','Class']

irisdata=pd.read_csv(url,names=names)

X=irisdata.iloc[:,3:4].values
#print(X)

y=(irisdata.iloc[:,4:]=='Iris-virginica').astype(np.int)
#print(y)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X,y)

X_new = np.linspace(0,3,1000).reshape(-1,1)
print(X_new)

y_prob = LR.predict_proba(X_new)
print(y_prob)
plt.plot(X_new, y_prob[:,1], "g-", label="virginica")
plt.show()

example=LR.predict(([[2.3]]))
#print(example)
'''
# Train a logistic regression classifier to predict whether a flower is iris virginica or not
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X,y)
example = clf.predict(([[2.6]]))
print(example)'''

# Using matplotlib to plot the visualization



# print(y)
# print(iris["data"])
# print(X)



