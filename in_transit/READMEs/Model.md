# GraphDRP
GraphDRP is a Graph Convolutional Networks for Drug Response Prediction.


## Structure
GraphDRP consists of two subnetworks that learn from input features of cancer cell lines (CCLs) and drugs to predict IC50, a dose-independent treatment response. The encoded feature representations by the two subnetworks are concatenated and passed through dense layers for the prediction of IC50. Each CCL is represented by a vector of 735 binary features including variant conding and copy number alterations. Each drug is represented with a graph molecular structure where nodes and edges represent, respectively, the atoms and bonds of the a molecule (each atom is represented by 78 features). The CCL subnetwork consists of three 1-D CNNs followed by dense layers. For the drug subnetworks, four different configurations of graph neural network (GNN) modules have been explored, including GCN, GAT, and GIN.


## Data sources
The primary data sources that have been used to construct datasets for model training and testing (i.e., ML data) include:
- GDSC version 6.0 (cell line and drug ids, treatment response, cell line omics data)
- PubChem (drug SMILES)


## Data and preprocessing
CCL omics data and treatment response data (IC50) were downloaded from the GDSC website. Refer to [Data](Data.md) for more info regarding the raw data provided with the original GraphDRP model repo and preprocessing scripts allowing to generate ML data for model training and testing.


## Evaluation
Three evaluation schemes were used for the analysis of prediction performance.

- Mixed set: cell lines and drugs can appear in train, validation, and test sets.
- Cell-blind: no overlap on cell lines in train, validation, and test sets.
- Drug-blind: no overlap on drugs in train, validation, and test sets. 


## URLs
- Original GitHub: https://github.com/hauldhut/GraphDRP
- IMPROVE GitHub: https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/develop
- Data: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/GraphDRP/


## Reference
Nguyen, T.-T. et al. Graph convolutional networks for drug response prediction. *IEEE/ACM Trans Comput Biol Bioinform*. Jan-Feb 2022;19(1):146-154.
