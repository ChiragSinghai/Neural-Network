import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# location of dataset on the internet
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#assign column names to dataset
names=['sepal-length','sepal-width','petal-length','petal-width','Class']

irisdata = pd.read_csv(url,names=names)
irisdata.head()

#assign data from first four columns to X variable
X=irisdata.iloc[:,0:4]

#Assign data from first fifth columns to y variable
y=irisdata.select_dtypes(include=[object])
y.head()
y.Class.unique()
np.array(['Iris-setosa','Iris-versicolor','Iris-virginica'],dtype=object)
le=preprocessing.LabelEncoder()
y = y.apply(le.fit_transform)
np.array([0,1,2],dtype=np.int64)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)
print(X_train)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test= scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(X_train,y_train.values.ravel())
predictions=mlp.predict(X_test)
print(predictions)
#from sklearn.metrics import classification_report,confusion_matrix
#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))





