# Uncovering Correlations Between Two UMAP Hyperparameters and the Input Dataset
## Abstract

Learning small high-dimensional image datasets can be 
challenging: while deep learning models struggle, because of
the limited data, simpler machine learning models can be 
slow, due to the high number of features. UMAP is a 
dimensionality reduction method that creates low
dimensional representations of the datasets, which can be 
used as input to simple models, reducing the computational 
time. However, finding the best hyperparameter values 
might require many tests. This thesis aims to reduce this 
guesswork by uncovering possible correlations between the 
dataset parameters and two UMAP hyperparameters. 9 
datasets have been embedded with different hyperparameter 
combinations and the KNN accuracy was computed to rank 
the embedding quality. A correlation between two 
hyperparameters was found. Datasets with lower number of 
features and/or lower silhouette followed such pattern most 
closely, with preferred hyperparameter value ranges. 
Moreover, training size did not seem to play a role in the 
choice of the hyperparameter values.

## Setup
Python3 and its packages from the requirements.txt files are required to run the code. 
```
pip install -r requirements.txt
```

## Running instructions
After all dependecies have been installed, run the [main.py](https://github.com/federicojv/Master_Thesis/main.py) script from command line with 
```
python3 main.py
```
This script relies the datasets folder to retrieve part of the used datasets. The rest of the datasets are downloaded from online sources. The raw output is saved in the results folder in the form of csv files, one per dataset plus an additional one called total, containing all the results .  
The graphs to better visualize the results can be created from the [analyse_results.ipynb](https://github.com/federicojv/Master_Thesis/analyse_results.ipynb) notebook.  
Due to the many embedding performed, the code took a total of 165 hours running on a Windows machine with an AMD Ryzen 5 1400 and 16 GBs of RAM.