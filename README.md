# Black-Box Testing of Deep Neural Networks through Test Case Diversity

This repository is a companion page for the following paper 
> "Black-Box Testing of Deep Neural Networks through Test Case Diversity".

This paper is implemented in python language with GoogleColab (It is an open-source and Jupyter based environment).


We have two main .ipynb files the first one `Testing_Experimnet.ipynb` is containing our emprical study and the second one `Fault definition.ipynb` is one of the required step for answering to two of our research questions (RQ2& RQ3).

`Testing_Experimnet.ipynb` contains the implementation of all diversity metrics (GD, STD, NCD) and all RQs.
DNNs faults are determined and saved for three models (LeNet1,LeNet5 and 12_Conv_layer) and two datasets (MNIST and Cifar10) by running `Fault definition.ipynb`.
`sadl11` folder contains the original code files for computing the LSC and DSC coverage metrics from 
> "Guiding Deep Learning System Testing using Surprise Adequacy" paper.

Requirements
---------------
You need to first install these Libraries:
  - `!pip install umap-learn`
  - `!pip install tslearn`
  - `!pip install hdbscan`

The code was developed and tested based on the following environment.

python 3.8

keras 2.7.0

Tensorflow 2.7.1

pytorch 1.10.0

torchvision 0.11.1

matplotlib

sklearn

tqdm

---------------
Here a documentation on how to use the replication material should be provided.

### Getting started

1. First, you need to upload the repo on your google drive and run the codes with https://colab.research.google.com/.
2.  The main code that you need to run is `Testing_Experimnet.ipynb`. This code covers all the datasets and models that we used in the paper, however if you want to replicate the results for LeNet5 model, you need to change two lines of the code in `sadl11/run.py` related to the loading model and selected layer.

comment out these lines : 1- `model = load_model("/content/drive/MyDrive/sadl11/model/model_mnist_LeNet5.h5")`                                                                                                                                      `layer_names = ["activation_13"]`
comment these lines :      `model= load_model("/content/drive/MyDrive/sadl11/model/model_mnist_LeNet1.h5")`
                           `layer_names = ["conv2d_1"]`
 

Repository Structure
---------------
This is the root directory of the repository. The directory is structured as follows:

    Replication-package
     .
     |
     |--- sadl11/model/                    Pre-trained models used in the paper
     |
     |--- RQ2-2/Correlation/               Random samples (60 subsets with sizes of 100,...,1000) to replicate the paper's results
     |
     |--- RQ3/                             The computation time of each metric for each samples               
  

Usually, replication packages should include:
* a [src](src/) folder, containing the entirety of the source code used in the study,
* a [data](data/) folder, containing the raw, intermediate, and final data of the study
* if needed, a [documentation](documentation/) folder, where additional information w.r.t. this README is provided. 

In addition, the replication package can include additional data/results (in form of raw data, tables, and/or diagrams) which were not included in the study manuscript.
