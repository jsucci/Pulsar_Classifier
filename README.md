# Pulsar_Classifier

Pre-requisites:

    Python 3 packages:
    -numpy
    -pandas
    -torch
    -tqdm
    -sklearn
    -lightning
    -torchmetrics
    -matplotlib


The projects works around a simple binary classification problem for pulsar observations using machine learning techniques.
In order to get the best performance the computations are made by the CUDA GPU (if none is available the scripts needs to be modified accordingly).
The projects uses two python scripts dataAug.py and pulsar.py. The first trains and adversary generative NN to perform data augmentation on the original dataset while the latter is responsible for the training and verification of the classfier.

The dataset can be found at this link : https://www.kaggle.com/datasets/firmanhasibuan1/pulsar-dataset (a copy of the .csv files are included in the repository).

In order to run the project must be executed first the dataAug.py script and once it has finished run the pulsar.py script.
In the "Figures" directory can be found the metrics of the classifier, its ROC and the generative NN error metric.



