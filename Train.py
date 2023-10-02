import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from d2l import torch as d2l
#定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1=nn.Linear(4,10)
        self.l2=nn.Linear(10,1)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=self.l2(x)
        return x
#读取数据并加载为Tensor形式、
Train_data=pd.read_csv("D:/tools/Python/pythonProject/Data/train1.csv")
Tensor_Train_data=torch.tensor(Train_data.iloc[:,:-1].values,dtype=torch.float32,device="cuda")
Labels=torch.tensor(Train_data.iloc[:,-1].values.reshape(-1,1),dtype=torch.float32,device="cuda")
#定义迭代器
train_iter=d2l.load_array((Tensor_Train_data,Labels),batch_size=4)
net=Net()
net.to(device="cuda")

#定义损失函数，这里使用均方误差。
loss=nn.MSELoss()
#定义优化器，这里使用随机梯度下降法。
optimizer=optim.Adam(net.parameters(),lr=0.0001)
for epoch in range(500):
    for x,y in train_iter:
        # 梯度清零
        optimizer.zero_grad()
        # 向前传播得到预测值
        Prediction = net.forward(x)
        # 计算损失函数
        L = loss(Prediction, y)
        # 反向传播，更改参数
        L.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 500, L.item()))

print(net.state_dict())