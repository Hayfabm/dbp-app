import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


from utils import create_dataset, one_hot_encoding


if __name__ == "__main__":

    # fix the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # eval parameters
    BATCH_SIZE = 128
    MAX_SEQ_LENGTH = 800

    # create test dataset
    sequences_test, labels_test = create_dataset(data_path="data/PDB2272.csv")  # 0.7848

    # encode sequences
    sequences_test_encoded = np.concatenate(
        [one_hot_encoding(seq, MAX_SEQ_LENGTH) for seq in sequences_test], axis=0,
    )  # (2272, 800, 20)

    # encode labels
    labels_test_encoded = to_categorical(
        labels_test, num_classes=2, dtype="float32"
    )  # (2272, 2)

    # load model
    model = load_model("checkpoint/model.15-0.78.hdf5")
    print(model.summary())

    test_loss, test_acc = model.evaluate(
        sequences_test_encoded, labels_test_encoded, batch_size=BATCH_SIZE
    )
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)
