import numpy as np
class Perceptron:
    def __init__(self,input_length,weights=None):
        if weights is None:
            self.weights=np.ones(input_length)*0.5
        else:
            self.weights=weights
    def unit_step_function(x):
        if x>0.35:
            return 1
        return 0
    def call(self,in_data):
        weighted_input=self.weights * in_data
        weighted_sum=weighted_input.sum()
        return Perceptron.unit_step_function(weighted_sum)
p=Perceptron(2,np.array([0.5,0.5]))
mat=[np.array([0,0]),np.array([0,1]),
     np.array([1,0]),np.array([1,1])]

for x in mat:
    y=p.call(np.array(x))
    print(x,y)
