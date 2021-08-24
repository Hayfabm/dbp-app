"""
WARNINGS: if you run the app locally and don't have a GPU you should choose device='cpu'
"""

from typing import Dict, List, Tuple

import pandas as pd
from keras.models import load_model
import numpy as np


class DBPApp:
    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.num_gpus = 0 if device == "cpu" else 1

        self.NATURAL_AA = "ACDEFGHIKLMNPQRSTVWY"
        self.max_seq_length = 600

        # NOTE: if you have issues at this step, please use h5py 2.10.0
        # by running the following command: pip install h5py==2.10.0
        self.model = load_model("checkpoint/model_0.hdf5")
        print(self.model.summary())

    def compute_scores(self, sequences_list: List[str]) -> List[float]:
        """Compute a score based on a user defines function.

        This function compute a score for each sequences receive in the input list.
        Caution :  to load extra file, put it in src/ folder and use
                   self.get_filepath(__file__, "extra_file.ext")

        Returns:
            ScoreList object
            Score must be a list of dict:
                    * element of list is protein score
                    * key of dict are score_names
        """

        scores_list = []
        for sequence in sequences_list:

            # adapt sequence size
            if len(sequence) > self.max_seq_length:
                sequence = sequence[: self.max_seq_length]
            else:
                sequence = sequence + "Z" * (self.max_seq_length - len(sequence))

            # encode sequence
            encoded_sequence = np.zeros((len(sequence), 20))
            for i, val in enumerate(sequence):
                if val in self.NATURAL_AA:
                    index = self.NATURAL_AA.index(val)
                    encoded_sequence[i][index] = 1
            model_input = np.expand_dims(encoded_sequence, 0)

            # forward pass throught the model
            model_output = self.model.predict(model_input)[0]

            scores_list.append(model_output[1])  # probability to bind
        return scores_list


def create_dataset(data_path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(data_path)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle the dataset
    return list(df["sequence"]), list(df["label"])


if __name__ == "__main__":
    # sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQ", "KALEE", "LAGYNIVATPRGYVLAGG"]

    # sequences, labels = create_dataset(data_path="data/PDB14189.csv")  # 0.7614
    sequences, labels = create_dataset(data_path="data/PDB2272.csv")  # 0.6558

    app = DBPApp("cpu")

    scores = app.compute_scores(sequences)

    # compute accuracy of the model
    acc = []
    for s, l in zip(scores, labels):
        if ((s >= 0.5) and (l == 1)) or ((s < 0.5) and (l == 0)):
            acc.append(1)
        else:
            acc.append(0)
    print(np.mean(acc))
