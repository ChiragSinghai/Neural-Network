import numpy as npp
import matplotlib.pyplot as plt
 
from sklearn.datasets import load_digits
digits = load_digits()
 
#After loading the dataset let's get familiar with what we have loaded in "digits".
#print(digits) 
print(len(digits.data[0]))
print((digits.target))
print('Image Data Shape' , digits.data.shape)
print("Label Data Shape", digits.target.shape)

for i in range(5):
    image=digits.data[i]
    label=digits.target[i]
    plt.imshow(npp.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Digities', fontsize = 20)
    plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(x_train, y_train)

predictions = LR.predict(x_test)
score = LR.score(x_test, y_test)
print("Accuracy is ",score*100,"%")
