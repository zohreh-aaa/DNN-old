# Black-Box Testing of Deep Neural Networks through Test Case Diversity

This repository is a companion page for the following paper 
> "Black-Box Testing of Deep Neural Networks through Test Case Diversity".

This paper is implemented in python language with GoogleColab (It is an open-source and Jupyter based environment).


We have two main .ipynb files the first one (Testing_Experimnet.ipynb) is containing our emprical study and the second one "Fault definition.ipynb" is one of the required step for answering to two of our research questions (RQ2& RQ3).


DNNs faults are determined and saved for three models (LeNet1,LeNet5 and 12_Conv_layer) and two datasets (MNIST and Cifar10) by running "Fault definition.ipynb".

Requirements for Fault definition part
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

n
---------------
Here a documentation on how to use the replication material should be provided.

### Getting started

1. Provide step-by-step instruction on how to use this repository, including requirements, and installation / script execution steps.

2. Code snippets should be formatted as follows.
   - `git clone https://github.com/S2-group/template-replication-package`

3. Links to specific folders / files of the repository can be linked in Markdown, for example this is a link to the [src](src/) folder.

Repository Structure
---------------
This is the root directory of the repository. The directory is structured as follows:

    template-replication-package
     .
     |
     |--- src/                             Source code used in the thesis / paper
     |
     |--- documentation/                   Further structured documentation of the replication package content
     |
     |--- data/                            Data used in the thesis / paper 
            |
            |--- additional_subfolder/     Subfolders should be further nested to increase readability                 
  

Usually, replication packages should include:
* a [src](src/) folder, containing the entirety of the source code used in the study,
* a [data](data/) folder, containing the raw, intermediate, and final data of the study
* if needed, a [documentation](documentation/) folder, where additional information w.r.t. this README is provided. 

In addition, the replication package can include additional data/results (in form of raw data, tables, and/or diagrams) which were not included in the study manuscript.
