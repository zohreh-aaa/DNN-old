# Black-Box Testing of Deep Neural Networks through Test Case Diversity

This repository is a companion page for the following paper 
> "Black-Box Testing of Deep Neural Networks through Test Case Diversity".

> Zohreh Aghababaeyan (uOttawa, Canada), Manel Abdellatif (uOttawa, Canada), Lionel Briand (uOttawa, Canada), Ramesh S (General Motors, USA), and Mojtaba Bagherzadeh (uOttawa, Canada)

This paper is implemented in python language with GoogleColab (It is an open-source and Jupyter based environment).


We have two main .ipynb files the first one `Testing_Experimnet.ipynb` contains our emprical study and the second one `Fault definition.ipynb` is one of the required step for answering to two of our research questions (RQ2 & RQ3).

`Testing_Experimnet.ipynb` contains the implementation of all diversity metrics (GD, STD, NCD) and all RQs.

DNNs faults are determined and saved for three models (LeNet1,LeNet5 and 12_Conv_layer) and two datasets (MNIST and Cifar10) by running `Fault definition.ipynb`.

This is the workflow of this part.

![image](https://user-images.githubusercontent.com/58783738/146564128-ab4ef712-635a-489e-b8a7-e764ae1972a8.png)

Part of the implementation is from [1] and [2] 

* [sadl11](sadl11/) folder contains some parts of [1] for computing the LSC and DSC coverage metrics.

Requirements
---------------
You need to first install these Libraries:
  - `!pip install umap-learn`
  - `!pip install tslearn`
  - `!pip install hdbscan`

The code was developed and tested based on the following environment:

- python 3.8
- keras 2.7.0
- Tensorflow 2.7.1
- pytorch 1.10.0
- torchvision 0.11.1
- matplotlib
- sklearn

---------------
Here a documentation on how to use this replication package.

### Getting started

1. First, you need to upload the repo on your Google drive and run the codes with https://colab.research.google.com/.
2. The main code that you need to run is `Testing_Experimnet.ipynb`. This code covers all the datasets and models that we used in the paper, however if you want to replicate the results for LeNet5 model, you need to change two lines of the code in `sadl11/run.py` that are related to the loading model and the selected layer.
To do so please:

Change these lines :

`model = load_model("/content/drive/MyDrive/sadl11/model/model_mnist_LeNet5.h5")`

`layer_names = ["activation_13"]`

With these two lines :     

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

<img width="929" alt="Diversity" src="https://user-images.githubusercontent.com/58783738/146585778-6dd7c17c-c8f8-4c6c-bda3-316e20e871b9.png">


-->Outcome:  GD and STD performed well in determining data diversity for all datasets examined. Our experiments exclude NCD because it does not measure data diversity accurately in our context.


RQ2: How does diversity relate to fault detection?

It is our goal to find out whether higher diversity results in better fault detection. For this purpose, we randomly select, with replacement, 60 samples of sizes 100, 200, 300, 400, 1000. For each sample, we calculate the diversity scores and the number of faults. Finally, we calculate the correlation between diversity scores and the number of faults.

Outcome:

DNN faults and GD have a moderately positive correlation. GD is more significantly correlated to faults than STD.  


RQ3: How does coverage relate to fault detection?

We aim to study the correlation between state-of-the-art coverage criteria and faults in DNNs.

Outcome:

In general, there is no significant correlation between DNN coverage and faults for natural dataset. LSC coverage showed a moderate positive correlation in only one configuration.

(RQ2 and RQ3 results)

![image](https://user-images.githubusercontent.com/58783738/146563862-579f5227-450d-432d-a1ae-9c27c10f1781.png)


RQ4: How do diversity and coverage metrics perform in terms of computation time?

In this research question, we aim to compare how the calculation times scale with the size of test sets for the selected diversity and coverage metrics.

<img width="914" alt="Computation" src="https://user-images.githubusercontent.com/58783738/146585618-ed8d772c-30f8-4870-a7ef-f5ce283703d4.png">

--> Outcome: Both diversity and coverage metrics are not computationally expensive for subsets with size of lower than 500. However, the selected diversity metrics outperform coverage metrics.


RQ5. How does diversity relate to coverage?

It aims to assess whether diverse input sets contribute to an increase in coverage of DNN models, and conversely, whether increasing coverage will increase diversity.


![image](https://user-images.githubusercontent.com/58783738/146586359-b531c770-354e-4941-a6d5-8006b5f1dcb5.png)


--> Outcome: In general, there is no significant correlation between diversity and coverage in DNN models.

Notes
-----

1- We used the same recommended settings of LSC and DSC hyperparameters (upper bound, lower bound, number of buckets, etc.) as in the original paper for the diferent models and datasets in our experiments.

2- For the LeNet1 model, we use the same setting as LeNet5 model.

3- For speed-up, you can use GPU-based tensorflow by changeing the Colab Runtime.

References
-----
1- [Surprise Adequacy](https://github.com/coinse/sadl)

2- [DBCV](https://github.com/christopherjenness/DBCV)
