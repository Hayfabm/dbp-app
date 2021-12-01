# deepchain.bio DNA-binding Proteins app #

## Install dbp conda environment ##

From the root of this repo, run
`conda env create -f environment.yaml`

Make sure you've tensorflow==2.5.0 installed

Follow this [tutorial](https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras#step-5-monitor-your-tensorflow-keras-training-in-neptune) to make neptune logger works


## Overview ##
As a part of the protein family, DNA-binding proteins play an important role in DNA replication, DNA methylation, gene expression and other biological processes. Due to the importance of DBPs, it is highly desirable to develop effective methods to identify DBPs. At present, some experimental techniques, such as filter binding assays, X-ray crystallography [[1]](https://pubmed.ncbi.nlm.nih.gov/11229439/) genetic analysis [[2]](https://pubmed.ncbi.nlm.nih.gov/12837780/), etc., are developed for identifying DBPs. However, experimental methods are both costly and time-consuming. Meanwhile, more and more protein sequences have exploded with efficient next-generation sequencing techniques. Therefore, it is an important research topic to develop fast and effective computational methods to handle such large-scale protein sequence data.

## Goals ##
1. Prediction of DNA binding proteins based on contextual features in amino acid sequence. 
2. Deep learning based hotspot prediction in DNA-binding proteins.

## Model architecture ##
![Logo](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0225317.g005&type=large)


## Datasets ##
* [**PDB14189**](https://github.com/hfuulgb/PDB-Fusion/tree/main/DNA) dataset composed of 7,129 DBPs (positive samples) and 7,060 non-DBPs (negative samples). 
* [**PDB2272**](https://github.com/hfuulgb/PDB-Fusion/tree/main/DNA) dataset contained 1,153 DBPs and 1,119 non-DBPs. 
## References ##
* Author paper Li G, Du X, Li X, Zou L, Zhang G, Wu Z. 2021. [Prediction of DNA binding proteins using local features and long-term dependencies with primary sequences based on deep learning](https://doi.org/10.7717/peerj.11262).
* Author GitHub repository [hfuulgb/PDBP-Fusion](https://github.com/hfuulgb/PDB-Fusion).


