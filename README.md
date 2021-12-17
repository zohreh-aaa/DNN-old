# Black-Box Testing of Deep Neural Networks through Test Case Diversity

This repository is a companion page for the following paper 
> "Black-Box Testing of Deep Neural Networks through Test Case Diversity".

This paper is implemented in python language with GoogleColab (It is an open-source and Jupyter based environment).


We have two main .ipynb files the first one `Testing_Experimnet.ipynb` is containing our emprical study and the second one `Fault definition.ipynb` is one of the required step for answering to two of our research questions (RQ2& RQ3).

`Testing_Experimnet.ipynb` contains the implementation of all diversity metrics (GD, STD, NCD) and all RQs.
DNNs faults are determined and saved for three models (LeNet1,LeNet5 and 12_Conv_layer) and two datasets (MNIST and Cifar10) by running `Fault definition.ipynb`.

* [sadl11](sadl11/) folder contains the original code files for computing the LSC and DSC coverage metrics from following paper.
> "Guiding Deep Learning System Testing using Surprise Adequacy"

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

---------------
Here a documentation on how to use the replication material should be provided.

### Getting started

1. First, you need to upload the repo on your google drive and run the codes with https://colab.research.google.com/.
2. The main code that you need to run is `Testing_Experimnet.ipynb`. This code covers all the datasets and models that we used in the paper, however if you want to replicate the results for LeNet5 model, you need to change two lines of the code in `sadl11/run.py` related to the loading model and selected layer.
To do so please:

Comment out these lines :

`model = load_model("/content/drive/MyDrive/sadl11/model/model_mnist_LeNet5.h5")`

`layer_names = ["activation_13"]`

And comment these lines :     

`model= load_model("/content/drive/MyDrive/sadl11/model/model_mnist_LeNet1.h5")`

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
  

Research Questions
---------------
Our experimental evaluation answers the research questions below.

RQ1: To what extent are the selected diversity metrics (GD, STD and NCD) good measures for qualifying an input set? 
Our objective is to evaluate the reliability of the selected diversity metrics for measuring the real diversity of an image input set in terms of its features, in a controlled manner. The RQ1 algorithm and related plots are available in the paper.
Outcome:
GD and STD performed well in determining data diversity for all datasets examined. Our experiments exclude NCD because it does not measure data diversity accurately in our context.

RQ2: How does diversity relate to fault detection?

It is our goal to find out whether higher diversity results in better fault detection. For this purpose, we randomly select, with replacement, 60 samples of sizes 100, 200, 300, 400, 1000. For each sample, we calculate the diversity scores and the number of faults. Finally, we calculate the correlation between diversity scores and the number of faults.
![image](https://user-images.githubusercontent.com/58783738/146548737-ffd224dd-b1e3-46b1-ad0c-c41b0532aae9.png)

Outcome:
DNN faults and GD have a moderately positive correlation. GD is more significantly correlated to faults than STD.  


RQ3: How does coverage relate to fault detection?

We aim to study the correlation between state-of-the-art coverage criteria and faults in DNNs.

![image](https://user-images.githubusercontent.com/58783738/146548567-20a248d2-37ff-4e95-9268-d4db00a78493.png)


Outcome:
In general, there is no significant correlation between DNN coverage and faults for natural dataset. LSC coverage showed a moderate positive correlation in only one configuration.

additional data/results (in form of raw data, tables, and/or diagrams) which were not included in the study manuscript.
