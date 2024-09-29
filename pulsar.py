import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset , DataLoader
import lightning as lg
import torchmetrics
from torchmetrics import classification as cf
import matplotlib.pyplot as plt

df = pd.read_csv("DataSet/augmented.csv")
x=df.drop(columns=["Class"])
#print(x)
del x[x.columns[0]]
y=df["Class"]


#df_train = pd.read_csv("train.csv")
#x_train=df_train.drop(columns=["Class","id"])
#y_train=df_train["Class"]

#df_test = pd.read_csv("test.csv")
#x_test=df_test.drop(columns=["Class","id"])
#y_test=df_test["Class"]


class mlp(lg.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.auroc = cf.AUROC(task="binary")
        self.roc = cf.ROC(task="binary")
        self.train_acc = cf.Accuracy(task="binary")
        self.metricValues = []
        self.collection = torchmetrics.MetricCollection(cf.Accuracy(task="binary"),cf.Specificity(task="binary"),cf.Recall(task="binary"))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(8,16)
        #self.fc1 = nn.Linear(7,16)
        self.fc2 = nn.Linear(16,64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(128,128)
        self.fc5 = nn.Linear(128,64)
        self.fc6 = nn.Linear(64,16)
        self.fc7 = nn.Linear(16,1)
        self.criterion = nn.BCELoss()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        return optimizer
        
    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        x=self.relu(x)
        x=self.fc4(x)
        x=self.relu(x)
        x=self.fc5(x)
        x=self.relu(x)
        x=self.fc6(x)
        x=self.relu(x)
        x=self.fc7(x)
        x=self.sigmoid(x)
        return x
    
    def training_step(self,batch):
        x,y = batch
        x = x
        y = y.unsqueeze(1)
        pred = self.forward(x)
        loss = self.criterion(pred,y)
        self.train_acc.update(pred,y)
        return loss
        
    def validation_step(self,batch):
        x,y = batch
        x = x
        y = y.unsqueeze(1)
        pred = self(x)
        self.collection.update(pred,y)
        self.auroc.update(pred,y)
        self.roc.update(pred,y.to(dtype=int))

    def on_validation_epoch_end(self):
        cc =self.collection.compute()
        self.log("val_recall",cc["BinaryRecall"])
        self.log("val_spec",cc["BinarySpecificity"])
        self.log("val_acc",cc["BinaryAccuracy"])
        self.log("val_auroc",self.auroc.compute())
        self.metricValues.append(cc)
        self.collection.reset()
        self.auroc.reset()

    def on_train_epoch_end(self):
        self.log("Train_acc",self.train_acc.compute())
        self.train_acc.reset()

    def on_train_end(self):
        fig,ax = self.collection.plot(val=self.metricValues,together=True)
        fig.savefig("Figures/MLP_Metrics.png")
        fig2,ax = self.roc.plot()
        fig2.savefig("Figures/MLP_ROC.png")

class Pulsar_dataset(Dataset) :
    def __init__(self,x,y):
        self.x = torch.tensor(x.values).to(dtype=torch.float32)
        self.y = torch.tensor(y.values).to(dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

net  = mlp()
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

#x_aug = pd.read_csv("culo.csv")
#del x_aug[x_aug.columns[0]]
#y_aug = pd.Series([1]*len(x_aug))

#print(y_train)
#x_train = pd.concat([x_train,x_aug])
#y_train = pd.concat([y_train,y_aug])
#print(y_train)

ds_train = Pulsar_dataset(x_train,y_train)
ds_test = Pulsar_dataset(x_test,y_test)

dl_train = DataLoader(ds_train, shuffle = True,batch_size=128,num_workers=15)
dl_test = DataLoader(ds_test, shuffle = False,batch_size=128,num_workers=15)

trainer = lg.Trainer(max_epochs=10)
trainer.fit(net,dl_train,dl_test)