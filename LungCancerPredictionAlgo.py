import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
df=pd.read_csv('Lung_cancer.csv')
df=df[['Age', 'Smokes', 'Alkhol', 'Result']]
X=df.iloc[:,0:3]
#print(X)
y=df['Result']
def work(size,layer,iters):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = size)
    #print(df)
    #print(X_test,y_test)

    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test= scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    mlp=MLPClassifier(hidden_layer_sizes=layer,max_iter=iters)
    mlp.fit(X_train,y_train.values.ravel())
    predictions=mlp.predict(X_test)
    #print(predictions)

    from sklearn.metrics import classification_report,confusion_matrix
    mat=confusion_matrix(y_test,predictions)
    try:
        accuracy=(mat[0][0]+mat[1][1])/y_test.shape[0]
        return accuracy
    except:
        print(mat,y_test.shape[0])
    #print(type(classification_report(y_test,predictions)))
layers=[(8,8,8,8),(10,10,10,10),(9,9,9,9),(9,9,9),(8,8,8),(10,10,10)]
test_size=[0.1,0.15,0.2,0.25,0.3,0.35]
iteration=[500,600,700,800,900,1000,1100]
count=0
column_name=['Test_size','epoch','Hidden_layers','accuracy']
data=pd.DataFrame(columns=column_name,index=range(253))

for i in layers:
    for j in test_size:
        for k in iteration:
            acc=work(j,i,k)
            data.loc[count]=[j,k,len(i),acc]
            count+=1
print(data)
data.to_csv('Best_Solutions.xlsx')
            
            
print(count)
    
    

    
''''''

