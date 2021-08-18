import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import tensorflow as tf
from numpy import array
from keras.optimizers import *
from keras import layers, models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, SeparableConv1D
from keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
    ModelCheckpoint,
)
from keras.layers import LSTM, Bidirectional, Dense, Embedding, Input, concatenate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.utils import plot_model
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers.normalization import BatchNormalization

AA_MAPPING = {
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "E": 6,
    "Q": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "X": 21,
    "B": 22,
    "U": 23,
    "O": 24,
    "Z": 25,
}


def create_dataset():
    # Read data {Positives samples}
    P = pd.read_csv("data/PDB14189_P.csv")
    # Split positive data in 80:10:10 for train:valid:test dataset
    train_size = 0.8
    # In the first step we will split the data in training and remaining dataset
    P_train, P_rem = train_test_split(P, train_size=0.8, random_state=1024)
    # Now since we want the valid and test size to be equal (10% each of overall data).
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    P_valid, P_test = train_test_split(P_rem, test_size=0.5, random_state=1)

    # Read data {Negative samples}
    N = pd.read_csv("data/PDB14189_N.csv")
    # Let's say we want to split negative data in 80:10:10 for train:valid:test dataset
    train_size = 0.8
    # In the first step we will split the data in training and remaining dataset
    N_train, N_rem = train_test_split(N, train_size=0.8, random_state=1024)
    # Now since we want the valid and test size to be equal (10% each of overall data).
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    N_valid, N_test = train_test_split(N_rem, test_size=0.5, random_state=1)

    # Get balanced Data
    train_C = pd.concat([P_train, N_train])
    valid_C = pd.concat([P_valid, N_valid])
    test_C = pd.concat([P_test, N_test])

    # shuffle the DataFrame rows
    train_C = train_C.sample(frac=1, random_state=1)
    valid_C = valid_C.sample(frac=1, random_state=1)
    test_C = test_C.sample(frac=1, random_state=1)

    x_train = train_C["SequenceID"]
    x_valid = valid_C["SequenceID"]
    x_test = test_C["SequenceID"]

    y_train = train_C["label"]
    y_valid = valid_C["label"]
    y_test = test_C["label"]

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def build_model(top_words, embedding_size, maxlen, pool_length):
    model = Sequential()
    model.add(Embedding(top_words, embedding_size, input_length=maxlen))
    model.add(
        Convolution1D(
            64,
            8,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer="random_uniform",
            name="convolution_1d_layer1",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_length))
    model.add(Dropout(0.2))
    model.add(
        Convolution1D(
            64,
            8,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer="random_uniform",
            name="convolution_1d_layer2",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_length))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(128, activation="sigmoid"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))

    return model


if __name__ == "__main__":
    # Create Dataset
    x_train, x_valid, x_test, y_train, y_valid, y_test = create_dataset()
    print("number of training data:", len(x_train))

    # Embedding
    top_words = len(AA_MAPPING.keys())
    maxlen = 800
    embedding_size = 28
    # Convolution
    pool_length = 3
    # TrainingSet
    batch_size = 4
    epochs = 200
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

    # Build Model
    model = build_model(top_words, embedding_size, maxlen, pool_length)
    print(model.summary())
    # Compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # integer encode the sequences
    encoded_docs = [[AA_MAPPING[aa] for aa in seq] for seq in x_train]
    encoded_docs2 = [[AA_MAPPING[aa] for aa in seq] for seq in x_valid]
    encoded_docs3 = [[AA_MAPPING[aa] for aa in seq] for seq in x_test]
    X_train = sequence.pad_sequences(
        encoded_docs, maxlen=maxlen, padding="post", value=0.0
    )
    X_valid = sequence.pad_sequences(
        encoded_docs2, maxlen=maxlen, padding="post", value=0.0
    )
    X_test = sequence.pad_sequences(
        encoded_docs3, maxlen=maxlen, padding="post", value=0.0
    )

    # Define y
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2, dtype="float32")
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=2, dtype="float32")
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2, dtype="float32")

    tf.config.experimental_run_functions_eagerly(True)
    # Fit the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=20,
        verbose=1,
        validation_data=(X_valid, y_valid),
    )
    # callbacks=[reduce_lr, early_stopping])

