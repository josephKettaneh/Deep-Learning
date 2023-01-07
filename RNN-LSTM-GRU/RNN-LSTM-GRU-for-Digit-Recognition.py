# Thierry Paquet C 2019-2021
# University of Rouen Normandie
# Master SID
# 
import keras
from keras.datasets import mnist
#from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
from keras.layers import Dense, Input, LSTM, GRU, SimpleRNN
from keras.optimizers import RMSprop
# tensorflow.from keras.optimizers import RMSprop
from keras.engine import Model
#from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint
#from tensorflow.keras.callbacks import ModelCheckpoint


import numpy as np
import matplotlib.pyplot as plt

def mlp_network(nb_features):
    """ Multilayer Perceptron
    nb_features = length of the input vector
    """
    inputs = Input(name='input', shape=(nb_features,))
    layer = Dense(64, activation='relu')(inputs)
    layer = Dense(64, activation='relu')(layer)
    predictions = Dense(10, activation='softmax')(layer)
    network = Model(inputs=inputs, outputs=predictions)
    network.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return network

def rnn_classif_network(T, D, RNN_TYPE):
    """ Recurrent Neural Network 
    T = sequence length
    D = number of features per frame
    """

    inputs = Input(name='input', shape=[T, D])
    if RNN_TYPE == 'LSTM':
        layer = LSTM(100, return_sequences=True)(inputs)
        layer = LSTM(100, return_sequences=False)(layer)
        predictions = Dense(10, activation='softmax')(layer)
    elif RNN_TYPE == 'RNN':
        layer = SimpleRNN(100, return_sequences=False)(inputs)
        #layer = SimpleRNN(100, return_sequences=False)(inputs)
        predictions = Dense(10, activation='softmax')(layer)

    network = Model(inputs=inputs, outputs=predictions)
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network

if __name__ == '__main__':

    # x are of dimension nb_data 28 x 28 which is the size of each image
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    # see the  11th image 
    plt.imshow(x_train[11,:,:], cmap='gray')
    plt.show()
    
    nb_classes = 10
    nb_train, T, D = x_train.shape
    nb_test, T, D = x_test.shape
    
    # keep somme data for validation
    END_TRAINING = 54000
    START_VALID = 54000
    
    x_valid = x_train[START_VALID:,:,:] 
    y_valid = y_train[START_VALID:] 

    x_train = x_train[:END_TRAINING,:,:] 
    y_train = y_train[:END_TRAINING] 
    
    ###########################################################################
    # one hiot encoding of the ground truth
    # Y-train [20000,10]]
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test, nb_classes)
    y_valid = to_categorical(y_valid, nb_classes)
    
    #model_name = "LSTM_50.epoch{epoch:02d}.hdf5"
    model_name = 'Best_LSTM_100_100.hdf5'
    ###########################################################################
    # declare and compile the network 
    net = rnn_classif_network(T, D,'LSTM')
    # set a checkpoint for saving the best valid model over the training iterations 
    checkpoint = ModelCheckpoint(filepath=model_name, 
                             monitor ='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
    ###########################################################################
    # Training the network with a batch size 512 using the specified validation dataset
    # nb_epoch = 40
    # returns the learning history  
    Enregistrement = net.fit(x_train, y_train,
                             batch_size=512,
                             validation_data=(x_valid,y_valid),
                             epochs=40,
                             callbacks=[checkpoint])
    print(Enregistrement.history.keys())
    ###########################################################################
    # plot the training history
    loss_train = Enregistrement.history['loss']
    loss_valid = Enregistrement.history['val_loss']    
    metric_train = Enregistrement.history['accuracy']    
    metric_valid = Enregistrement.history['val_accuracy']    
        
    plt.plot(loss_train,"b:o", label = "loss_train")
    plt.plot(loss_valid,"r:o", label = "loss_valid") 
    
    plt.title("Loss over training epochs")
    plt.legend()
    plt.show()
  
    plt.plot(metric_train,"b:o", label = "accuracy_train")
    plt.plot(metric_valid,"r:o", label = "accuracy_valid") 
    plt.title("Accuracy over training epochs")
    plt.legend()
    plt.show()
    ###########################################################################
    #    Let's load the best saved model and test on the test dataset
    model = keras.models.load_model(model_name)
    loss_test = net.evaluate(x_test, y_test)
    #print("\nTEST LOSS AND ACCURACY = ", loss_test)
    

