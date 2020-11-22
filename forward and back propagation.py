import numpy as np
X=np.array([[1,0,0,1],[0,0,1,1],[1,1,1,1]])
y=np.array([[1],[0],[1]])
def sigmoid(x):
    print(x)
    return 1/(1+np.exp(-x))
def derivatives_sigmoid(x):
    print(x)
    return x*(1-x)
#variable init
epoch=1#setting training iteration
lr=0.1#setting learning rate
inputlayer_neurons=X.shape[1] # number of feat
hiddenlayer_neurons=2# number of hidden layer
output_neurons=1 # no. of neurons at output layer
print('inputlayerneurons',inputlayer_neurons)
print('shape',X.shape)
# weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
print('wh',wh)
print('bh',bh)
print('wout',wout)
print('bout',bout)
for i in range(epoch):
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1+bh
    print('hidden_layer_input',hidden_layer_input)
    hiddenlayer_activations=sigmoid(hidden_layer_input)
    print('hiddenlayer_activations',hiddenlayer_activations)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    print('output_layer_input1',output_layer_input1)
    output_layer_input=output_layer_input1+bout
    output=sigmoid(output_layer_input)
    print('output',output)
    '''
    E=y-output
    print(E)
    slope_output_layer=derivatives_sigmoid(output)
    print(slope_output_layer)
    slope_hidden_layer=derivatives_sigmoid(hiddenlayer_activations)
    d_output=E*slope_output_layer
    Error_at_hidden_layer=d_output.dot(wout.T)
    d_hiddenlayer=Error_at_hidden_layer*slope_hidden_layer
    wout+=hiddenlayer_activations.T.dot(d_output)*lr
    bout+=np.sum(d_output,axis=0,keepdims=True)*lr
    wh+=X.T.dot(d_hiddenlayer)*lr
    bh+=np.sum(d_hiddenlayer,axis=0,keepdims=True)*lr
    print('hey:',output)

    
print(np.dot(X,wh))
    '''
