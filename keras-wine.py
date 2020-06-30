# Ryan Haas
# CSCI 450
# Lab 6 - Neural Network Lab

# Analysis
# So while exploring how to do this project I understood the basic layer process
# and how you constructed the neural net. I hated, however, how much black magic
# was involved in the hidden layers. I tried increasing the number of nodes,
# adding more layers, removing layers, etc. and it affected the results in seemingly
# random ways. At times if I added a layer it could drop the success rate by an insane
# amount, but removing a layer could increase the percent just as much. I pretty much
# just started guessing random numbers for nodes and layers until I got something
# that was reasonable. I ultimately found that there is not consistent way to predict
# how well a neural net will perform when adding or removing layers or nodes.

# System imports
from numpy import loadtxt
from numpy import zeros
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

def get_model():
    # Load training data
    data = loadtxt('wine_train.csv', delimiter=',')
    # Data starts after the first column, so split off the first column
    inData = data[:,1:]
    # The first column is what the output should be, so store that
    output = data[:,0:1]

    # Convert the output to categorical
    hotOutput = to_categorical(output)[:,1:]

    # Normalize training data
    scaler = MinMaxScaler((1,3))
    scaler.fit(inData)
    inData = scaler.transform(inData)

    # Define model/Add layers
    model = Sequential()
    model.add(Dense(8, input_dim=13, activation='relu')) 
    model.add(Dense(4, activation='selu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Fit model
    model.fit(inData, hotOutput, epochs=150, batch_size=10, verbose=0)

    # Return the model
    return model
