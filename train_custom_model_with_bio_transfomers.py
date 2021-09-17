"""DBP training script for custom model with one hot encoding"""
import datetime
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Flatten,
    Convolution1D,
    MaxPooling1D,
    BatchNormalization,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
)
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback


from utils import create_dataset, one_hot_encoding

from biotransformers import BioTransformers


def build_model(top_words, maxlen, pool_length):
    """PDBP-fusion model
    Combined CNN and Bi-LSTM, to predict DNA binding proteins
    """
    custom_model = Sequential(name="PDBPFusion")
    # (2272, 800, 20) <- one hot sequence encoding
    custom_model.add(InputLayer(input_shape=(maxlen, top_words)))
    # (2272, 800, 20) <- still the same one hot sequence encoding
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
    # init neptune logger
    run = neptune.init(
        project="sophiedalentour/DBP-APP",
        tags=["list-of", "tags", "goes-here", "as-list-of-strings"],
    )

    # set the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # set amino acids to consider
    CONSIDERED_AA = "ACDEFGHIKLMNPQRSTVWY"

    # embedding and convolution parameters
    MAX_SEQ_LENGTH = 800
    VOCAB_SIZE = len(CONSIDERED_AA)
    POOL_LENGTH = 3

    # training parameters
    BATCH_SIZE = 128
    NUM_EPOCHS = 2000
    SAVED_MODEL_PATH = (
        "logs/model_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".hdf5"
    )
    TRAIN_SET = "data/PDB14189.csv"
    TEST_SET = "data/PDB2272.csv"

    BIOTF_MODEL = "protbert"
    BIOTF_POOLMODE = "mean"
    BIOTF_BS = 2

    # save parameters in neptune
    run["hyper-parameters"] = {
        "encoding_mode": "one hot",
        "seed": SEED,
        "considered_aa": CONSIDERED_AA,
        "max_seq_length": MAX_SEQ_LENGTH,
        "vocab_size": VOCAB_SIZE,
        "pool_length": POOL_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "saved_model_path": SAVED_MODEL_PATH,
        "train_set": TRAIN_SET,
        "test_set": TEST_SET,
    }

    # create train dataset
    sequences_train, labels_train = create_dataset(data_path=TRAIN_SET)

    # create test dataset
    sequences_test, labels_test = create_dataset(data_path=TEST_SET)

    # encode sequences
    sequences_train_encoded = np.concatenate(
        [
            one_hot_encoding(seq, MAX_SEQ_LENGTH, CONSIDERED_AA)
            for seq in sequences_train
        ],
        axis=0,
    )  # (14189, 800, 20)
    sequences_test_encoded = np.concatenate(
        [
            one_hot_encoding(seq, MAX_SEQ_LENGTH, CONSIDERED_AA)
            for seq in sequences_test
        ],
        axis=0,
    )  # (2272, 800, 20)

    bio_trans = BioTransformers(backend=BIOTF_MODEL)

    sequences_train_embeddings = bio_trans.compute_embeddings(
        sequences_train, pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
    )[
        BIOTF_POOLMODE
    ]  # ?

    sequences_test_embeddings = bio_trans.compute_embeddings(
        sequences_test, pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
    )[
        BIOTF_POOLMODE
    ]  # ?

    # encode labels
    labels_train_encoded = to_categorical(
        labels_train, num_classes=2, dtype="float32"
    )  # (14189, 2)
    labels_test_encoded = to_categorical(
        labels_test, num_classes=2, dtype="float32"
    )  # (2272, 2)

    # build model
    model = build_model(VOCAB_SIZE, MAX_SEQ_LENGTH, POOL_LENGTH)
    print(model.summary())

    # compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )

    # define callbacks
    my_callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
        # EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1),
        ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
            filepath=SAVED_MODEL_PATH,
            save_best_only=True,
        ),
        NeptuneCallback(run=run, base_namespace="metrics"),
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

    run.stop()
