import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional, Embedding
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization

from utils import create_dataset, split_dataset


def build_model(top_words, embedding_size, maxlen, pool_length):
    """PDBP-fusion model
    Combined CNN and Bi-LSTM, to predict DNA binding proteins
    """
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

    # create dataset
    sequences, labels = create_dataset(data_path="data/PDB14189.csv")

    # split dataset
    (
        sequences_train,
        sequences_valid,
        sequences_test,
        labels_train,
        labels_valid,
        labels_test,
    ) = split_dataset(sequences, labels, train_size=0.8)

    # encode sequences
    sequences_train_encoded = sequence.pad_sequences(
        [[AA_MAPPING[aa] for aa in list(seq)] for seq in sequences_train],
        maxlen=MAX_SEQ_LENGTH,
        padding="post",
        value=0.0,
    )  # (11351, 800)
    sequences_valid_encoded = sequence.pad_sequences(
        [[AA_MAPPING[aa] for aa in seq] for seq in sequences_valid],
        maxlen=MAX_SEQ_LENGTH,
        padding="post",
        value=0.0,
    )  # (1419, 800)
    sequences_test_encoded = sequence.pad_sequences(
        [[AA_MAPPING[aa] for aa in seq] for seq in sequences_test],
        maxlen=MAX_SEQ_LENGTH,
        padding="post",
        value=0.0,
    )  # (1419, 800)

    # encode labels
    labels_train_encoded = tf.keras.utils.to_categorical(
        labels_train, num_classes=2, dtype="float32"
    )  # (11351, 2)
    labels_valid_encoded = tf.keras.utils.to_categorical(
        labels_valid, num_classes=2, dtype="float32"
    )  # (1419, 2)
    labels_test_encoded = tf.keras.utils.to_categorical(
        labels_test, num_classes=2, dtype="float32"
    )  # (1419, 2)

    # build model
    model = build_model(VOCAB_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH, POOL_LENGTH)
    print(model.summary())

    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    tf.config.experimental_run_functions_eagerly(True)

    # fit the model
    history = model.fit(
        sequences_train_encoded,
        labels_train_encoded,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=(sequences_valid_encoded, labels_valid_encoded),
    )
