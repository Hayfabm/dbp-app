import datetime
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import LSTM, Bidirectional, Embedding
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)


from utils import create_dataset


def build_model(top_words, embedding_size, maxlen, pool_length):
    """PDBP-fusion model
    Combined CNN and Bi-LSTM, to predict DNA binding proteins
    """
    custom_model = Sequential(name="PDBP-fusion model")
    custom_model.add(Embedding(top_words, embedding_size, input_length=maxlen))
    custom_model.add(
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
    custom_model.add(BatchNormalization())
    custom_model.add(MaxPooling1D(pool_size=pool_length))
    custom_model.add(Dropout(0.2))
    custom_model.add(
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
    custom_model.add(BatchNormalization())
    custom_model.add(MaxPooling1D(pool_size=pool_length))
    custom_model.add(Dropout(0.2))
    custom_model.add(Bidirectional(LSTM(32, return_sequences=True)))
    custom_model.add(Flatten())
    custom_model.add(Dense(128, activation="sigmoid"))
    custom_model.add(BatchNormalization())
    custom_model.add(Dropout(0.3))
    custom_model.add(Dense(2, activation="softmax"))

    return custom_model


if __name__ == "__main__":

    # fix the seed
    os.environ["PYTHONHASHSEED"] = str(42)
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # amino acids encoding
    NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
    SPECIAL_AA = "BJOUXZ"
    CONSIDERED_AA = NATURAL_AA + SPECIAL_AA
    AA_MAPPING = {aa: i + 1 for i, aa in enumerate(CONSIDERED_AA)}
    # embedding and convolution parameters
    VOCAB_SIZE = len(AA_MAPPING.keys())
    MAX_SEQ_LENGTH = 800
    EMBEDDING_SIZE = 28
    POOL_LENGTH = 3

    # training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 200

    # create train dataset
    sequences_train, labels_train = create_dataset(data_path="data/PDB14189.csv")

    # create test dataset
    sequences_test, labels_test = create_dataset(data_path="data/PDB2272.csv")

    # encode sequences
    sequences_train_encoded = sequence.pad_sequences(
        [[AA_MAPPING[aa] for aa in list(seq)] for seq in sequences_train],
        maxlen=MAX_SEQ_LENGTH,
        padding="post",
        value=0.0,
    )  # (14189, 800)
    sequences_test_encoded = sequence.pad_sequences(
        [[AA_MAPPING[aa] for aa in seq] for seq in sequences_test],
        maxlen=MAX_SEQ_LENGTH,
        padding="post",
        value=0.0,
    )  # (2272, 800)

    # encode labels
    labels_train_encoded = to_categorical(
        labels_train, num_classes=2, dtype="float32"
    )  # (14189, 2)
    labels_test_encoded = to_categorical(
        labels_test, num_classes=2, dtype="float32"
    )  # (2272, 2)

    # build model
    model = build_model(VOCAB_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH, POOL_LENGTH)
    print(model.summary())

    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    tf.config.experimental_run_functions_eagerly(True)

    # in order to see logs, please run this command: tensorboard --logdir logs/
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    my_callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1),
        ModelCheckpoint(
            filepath=log_dir + "/model.{epoch:02d}-{val_accuracy:.2f}.hdf5"
        ),
        TensorBoard(
            log_dir=log_dir, update_freq="epoch", profile_batch=0, histogram_freq=1
        ),
    ]

    # fit the model
    history = model.fit(
        sequences_train_encoded,
        labels_train_encoded,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=(sequences_test_encoded, labels_test_encoded),
        callbacks=my_callbacks,
    )
