import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TENSORFLOW_FLAGS'] = 'floatX=float32,device=cpu'

import cloudpickle as pickle
#import pipeline_predict
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import save_model, load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

if __name__ == '__main__':
    df = pd.read_csv("c.csv")
    # dropout the column that we don't need for training part
    data = df.drop(['Time'], axis=1)
    # squeeze the transaction amount
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    # splitting the data for train and test
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    X_train = X_train[X_train.Class == 0]
    X_train = X_train.drop(['Class'], axis=1)

    y_test = X_test['Class']
    X_test = X_test.drop(['Class'], axis=1)

    X_train = X_train.values
    X_test = X_test.values
    
    input_dim = X_train.shape[1]
    encoding_dim = 14
    # model layers
    input_layer = Input(shape=(input_dim, ))

    encoder = Dense(encoding_dim, activation="tanh", 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    # training params
    nb_epoch = 100
    batch_size = 32

    autoencoder.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="state/model.h5",
                                verbose=0,
                                save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True)

    history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        verbose=1,
                        callbacks=[checkpointer, tensorboard]).history
    # done training :)
