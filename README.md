# deepchain.bio DNA-binding Proteins app #

## Install dbp conda environment ##

From the root of this repo, run
`conda env create -f environment.yaml`

Make sure you've tensorflow==2.5.0 installed

Follow this [tutorial](https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras#step-5-monitor-your-tensorflow-keras-training-in-neptune) to make neptune logger works


## Overview ##
As a part of the protein family, DNA-binding proteins play an important role in DNA replication, DNA methylation, gene expression and other biological processes. Due to the importance of DBPs, it is highly desirable to develop effective methods to identify DBPs. At present, some experimental techniques, such as filter binding assays, X-ray crystallography (PMID: 11229439) genetic analysis (PMID: 12837780), etc., are developed for identifying DBPs. However, experimental methods are both costly and time-consuming. Meanwhile, more and more protein sequences have exploded with efficient next-generation sequencing techniques. Therefore, it is an important research topic to develop fast and effective computational methods to handle such large-scale protein sequence data.

## Goals ##
1. Prediction of DNA binding proteins based on contextual features in amino acid sequence. 
2. Deep learning based hotspot prediction in DNA-binding proteins.

## Specifications ##
“Our mission is to build an Apps “DNA-binding proteins” according to : 
- **(Li et al.,2021)** <https://peerj.com/articles/11262/#table-2>
- **(Hu et al., 2019)** <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6855455/ >
They applied a new deep learning model, named CNN-BiLSTM, to identify DNA- binding proteins. 
**CNN** :   Learn local features in amino acid sequence.
**Bi-LSTM** : Capture the contextual feature in both the forward and backward directions and thus achieve a better recognition effect for DNA-binding proteins.

## PLAN ##
### Data preparation ###
We will use already existing data PDB14189 from **Ma et al.,2016** (PMID: 27907159) and **Yu et al.,2006** (doi:10.1016/j.jtbi.2005.09.018)

The PDB14189 dataset consists of : 
 **7129** DBPs (positive samples) 
 **7060** non-DBPs (negative samples)

For obtaining binding proteins (positive samples)
- **30000** DNA-binding proteins VS NOW **32831** were collected from the Uniprot db (only reviewed).
- Proteins with lengths less than 50 AA and Proteins of more than 6000 AA were removed.
- Protein sequences including irregular AA (“x” and “Z”) were excluded.
- Redundant protein sequences were reduced using BLAST with a threshold of 40%.
-------> 7131 DBPs

For obtaining non-binding proteins (negative samples)
- **528,086** non-binding proteins VS NOW **519694** were processed according to the similarity criteria as the negative dataset.
- 67029 non-binding protein sequences were selected as the negative dataset (after cleaning)  
- 7131 non-binding proteins were randomly selected to balance with the positive dataset. 
Some proteins have been modified or removed due to the revision of the UniProt database. As a result, the benchmark data set that we used in this study consists of 7129 DBPs and 7060 non-DBPs.

**GOAL**: Balanced experiment dataset for model (done)
- Same number of (+) and (-) in each set (train, valid, test)
- Balanced length sequences (50-6000).

Dataset        | Positive samples | Negative samples | Total |
-------------- | ---------------- | ---------------- | ----- |
Original Data  | 7129             | 7060             | 14189 |
Training set   | 5648             | 5703             | 11351 |
Validation set | 706              | 713              | 1419  |
Testing set    | 706              | 713              | 1419  |

### Modal Framework ###
The deep learning model is composed of the following four parts: 
- Encoding
- Embedding 
- Convolutional layer
- Bi-LSTM layer

Goal
- Reduce time run 200---->5 epochs (for testing)
- Add callbacks (Early stopping, Reduce lr on plateau)
- Try using the Stratified K-Folds cross-validation
- Walk into biotransfomers strategy 


