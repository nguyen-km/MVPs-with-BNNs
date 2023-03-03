import pandas as pd
import numpy as np
df = pd.read_csv('/Volumes/GoogleDrive/My Drive/CSCI 5922 - Deep Learning/project/data/mvps.csv')

y_raw = df['Rank'].map(lambda x : x.strip('T')).values.reshape(-1,1)

# For single output, TRUE if player was "finalist" (1st, 2nd, or 3rd)
y=list()
for i in range(len(y_raw)):
    if(int(y_raw[i]) in [1, 2, 3]):
        y.append(1)
    else:
        y.append(0)

y = np.array(y).reshape(-1,1)

means=df.mean()
df = df.fillna(means)

X_raw = df.drop(['Player', 'Rank', 'First', 'Pts Won', 'Pts Max', 'Share', 'Tm'], axis=1)

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X_raw)

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X, y) # 30% split

class NeuralNetwork:
    def __init__(self, hidden_units=2, features=3, activation="sigmoid", reinit = False):
        #randomly initialize weights and bias
        self.__outputs = 1
        self.W1 = np.random.randn(features,hidden_units) #weight matrix, ncols x hidden units
        self.b = np.random.randn(hidden_units) # bias vector, hidden units x 1
        self.W2 = np.random.randn(hidden_units,self.__outputs) # second weight matrix, hidden units x 1
        self.__hidden_units = hidden_units
        self.H = None
        self.c = np.random.randn(self.__outputs)
        self.__Z = None
        self.__Z2 = None
    def get_weights(self):
        return({'W1' : self.W1, 'W2' : self.W2, 'b' : self.b})
    def activation(self, z, deriv = False):
        # sigmoid activation function
        return 1 / (1+np.exp(-z)) if (deriv==False) else np.multiply(z,1-z) 
    def train(self, X, y, eta = 0.1, epochs=1000):
        for i in range(epochs):
            y_hat = self.think(X)
            y = np.reshape(y, (len(y), self.__outputs)) # reshape
            error = y_hat - y # n x 1 (y^-y)
            y_hat_deriv = self.activation(y_hat, deriv=True) # n x 1 (y^)(1-y^)
            d_error = np.multiply(y_hat_deriv, error) # n x o (y^)(1-y^)(y^-y)
            H_deriv = self.activation(self.H, deriv=True) # n x h (Sig'(H))
            H_deriv_d_error_w2 = np.multiply((d_error @ self.W2.T), H_deriv)
            
            dW1 = X.T @ H_deriv_d_error_w2
            dW2 = self.H.T @ d_error
            db = np.multiply(H_deriv.T @ d_error, self.W2) # h x o
            dc = d_error.sum()
            
            self.W1 = self.W1 - (eta * dW1)
            self.W2 = self.W2 - (eta * dW2)
            self.b = np.squeeze(self.b.reshape(-1,1) - (eta*db))
            self.c = self.c - (eta*dc)
        return(y_hat)
            
    def think(self, X):
        self.__Z = (X @ self.W1) + self.b # n x h
        self.H = self.activation(self.__Z) # n x h
        self.__Z2 = (self.H @ self.W2) + self.c # (nxh)x(hx1) = n x 1
        y_hat = self.activation(self.__Z2)
        return y_hat
        
nn = NeuralNetwork(features = X_train.shape[1], hidden_units = 10)
nn_pred = nn.train(X_train, y_train)

#convert to 1s and 0s
y_pred = list()
for i in range(len(nn_pred)):
    if(nn_pred[i] >= 0.5):
        y_pred.append(1)
    else:
        y_pred.append(0)

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
f1_train = f1_score(y_pred, y_train)
acc_train = accuracy_score(y_pred, y_train)
print('Train F1:',f1_train,'\nTrain Accuracy:', acc_train)

nn_pred_test = nn.think(X_test)
y_pred = list()
for i in range(len(nn_pred_test)):
    if(nn_pred_test[i] >= 0.5):
        y_pred.append(1)
    else:
        y_pred.append(0)

f1= f1_score(y_pred, y_test)
acc = accuracy_score(y_pred, y_test)
print('Test F1:',f1,'\nTest Accuracy:', acc)
