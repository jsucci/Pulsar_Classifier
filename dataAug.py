#siccome siamo stronzi e il dataset è sbilanciato facciamo come si fa in laboratorio
#CREAIAMO DATI A CAZZO <3
#What is your sauce? the sauce is the time I did the fuck up <3

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


class adv_gen(lg.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.MSEmetric = torchmetrics.MeanSquaredError()
        self.MSEValues = []
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(1,16)
        self.fc2 = nn.Linear(16,64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(128,128)
        self.fc5 = nn.Linear(128,64)
        self.fc6 = nn.Linear(64,16)
        #self.fc7 = nn.Linear(16,7)
        self.fc7 = nn.Linear(16,8)
        self.criterion = nn.MSELoss()

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
        return x
    
    def training_step(self,batch):
        x,y = batch
        x = x
        y = y.unsqueeze(1)
        pred = self.forward(y)
        loss = self.criterion(pred,x)
        return loss
        
    def validation_step(self,batch):
        x,y = batch
        x = x
        y = y.unsqueeze(1)
        pred = self(y)
        self.MSEmetric.update(pred,x)

    def on_validation_epoch_end(self):
        mse = self.MSEmetric.compute()
        self.MSEValues.append(mse)
        self.MSEmetric.reset()

    def on_train_end(self):
        fig,ax = self.MSEmetric.plot(val=self.MSEValues)
        fig.savefig("Figures/GenError.png")
    

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
    

class Sampler:

    def __init__(self,net,sampleSize):
        self.net = net
        self.sampleSize = sampleSize
    
    def generateSample(self,hits):
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad =  False

        noise = torch.randn(hits.shape,device = hits.device)
        #samples = []
        hits.requires_grad = True
        for i  in range(self.sampleSize):
            noise.normal_(0,0.1)
            hits.data.add_(noise)
            hits.data.clamp_(min = 0,max = 1)
            output = -1.*net(hits)
            output.sum().backward()
            grad = hits.grad
            hits.grad.detach_()
            hits.grad.zero_()
            hits.grad.data.clamp(min = -0.1 ,max = +0.1 )
            hits.data.add_(grad.data)

            hits.data.clamp_(min = 0,max = 1)
            #samples.append(hits)

        for p in self.net.parameters():
            p.requires_grad =  True

        return (-1*output).detach_()

#df = pd.read_csv("Pulsar_cleaned.csv")
#x=df.drop(columns=["Class"])
#y=df["Class"]

df_train = pd.read_csv("DataSet/train.csv")
x_train=df_train.drop(columns=["Class","id"])
y_train=df_train["Class"]

#df_test = pd.read_csv("train.csv")
#x_test=df_test.drop(columns=["Class","id"])
#y_test=df_test["Class"]

xmax = x_train.max()
xmin = x_train.min()

x_train= (x_train-xmin)/(xmax-xmin)

net  = adv_gen()
x_train , x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.25)

ds_train = Pulsar_dataset(x_train,y_train)
ds_test = Pulsar_dataset(x_test,y_test)

dl_train = DataLoader(ds_train, shuffle = True,batch_size=128,num_workers=15)
dl_test = DataLoader(ds_test, shuffle = False,batch_size=128,num_workers=15)

trainer = lg.Trainer(max_epochs=5)
trainer.fit(net,dl_train,dl_test)

x_train = pd.concat([x_train,x_test])
y_train = pd.concat([y_train,y_test])
x_train = (x_train*(xmax-xmin)) + xmin

x_train["Class"] = y_train
n_gen = 100000

net = net.to(device="cuda") #comment if no CUDA available


sampler = Sampler(net,20)
for i in tqdm(range(n_gen)):
    gen_class = torch.rand(1,device = "cuda") #comment if no CUDA available
    #gen_class = torch.rand(1) #uncomment if no CUDA available
    d_gen = sampler.generateSample(gen_class)
    d_gen = (d_gen.cpu().numpy() *(xmax-xmin)) + xmin
    if gen_class < 0.5:
        gen_class = 0
    else:
        gen_class = 1

    d_gen = d_gen.to_frame().T
    d_gen["Class"] = gen_class
    x_train = pd.concat([x_train, d_gen])
    
x_train.to_csv("DataSet/augmented.csv")