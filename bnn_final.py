import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, confusion_matrix
import tensorflow_probability as tfp

# Data Cleaning
path = '/Volumes/GoogleDrive/My Drive/CSCI 5922 - Deep Learning/project/data/mvps.csv'
df = pd.read_csv(path)

df.head()

y_raw = df['Rank'].map(lambda x : x.strip('T')).values.reshape(-1,1) # Remove ties

# Code "Finalists" as positive class
y=list()
for i in range(len(y_raw)):
    if(int(y_raw[i]) in [1, 2, 3, 4, 5]):
        y.append(1)
    else:
        y.append(0)
y = np.array(y).reshape(-1,1)
y_raw[0:5]

df = df.fillna(df.fillna(df.mean())) #replace missing values with column averages

X_raw = df.drop(['Player', 'Rank', 'First', 'Pts Won', 'Pts Max', 'Share', 'Tm'], axis=1) # Remove irrelevant columns
X_raw.head()

# Standardize input matrix
X = StandardScaler().fit_transform(X_raw)
X[0:5]
numInputs = X.shape[1] 

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.25)
train_size = X_train.shape[0]

#ANN
mod_ann = keras.models.Sequential([
    layers.Dense(32, input_dim = numInputs, activation = "relu"),
    layers.Dense(16, input_dim = numInputs, activation = "relu"),
    layers.Dense(8, activation = 'relu'),
    layers.Dense(4, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')]
)
mod_ann.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
mod_ann.summary()

hist = mod_ann.fit(X_train, y_train, epochs = 500, batch_size = 64, validation_split = 0.25)

y_pred= mod_ann.predict(X_test) 
y_pred
y_pred_list = list()
for i in range(len(y_pred)):
    if(y_pred[i] >= 0.5):
        y_pred_list.append(1)
    else:
        y_pred_list.append(0)

a =f1_score(y_pred_list,y_test)
print('ANN F1 score is:', round(a, 4))
print(confusion_matrix(y_pred_list, y_test))

# BNN Code adapted from: https://keras.io/examples/keras_recipes/bayesian_neural_networks/

# Prior distributed as a multivariate Gaussian with mean 0, covariance I
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n)))]
    )
    return prior_model

# Define variational posterior weight distribution as multivariate Gaussian
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential([
        tfp.layers.VariableLayer(
            tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfp.layers.MultivariateNormalTriL(n),]
    )
    return posterior_model

hidden_units = [32, 16, 8, 4]
# Deterministic BNN
inputs = keras.layers.Input(shape=(numInputs))
features = keras.layers.BatchNormalization()(inputs)

# Create hidden layers with weight uncertainty using the DenseVariational layer.
for units in hidden_units:
    features = tfp.layers.DenseVariational(
        units=units,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1 / train_size,
        activation="relu",
    )(features)

# The output is deterministic: a single point estimate.
outputs = layers.Dense(units=1, activation = "sigmoid")(features)
bnn = keras.Model(inputs=inputs, outputs=outputs)

bnn.summary()

bnn.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
hist_bnn = bnn.fit(X_train, y_train, epochs = 500, batch_size = 64, validation_split = 0.25, verbose = False)
y_pred_bnn = bnn.predict(X_test) # returns prediction outputs as softmax

y_pred_list_bnn = list()
for i in range(len(y_pred_bnn)):
    if(y_pred_bnn[i] >= 0.5):
        y_pred_list_bnn.append(1)
    else:
        y_pred_list_bnn.append(0)

a =f1_score(y_pred_list_bnn,y_test)
print('BNN F1 score is:', round(a, 4))
confusion_matrix(y_pred_list_bnn, y_test)



