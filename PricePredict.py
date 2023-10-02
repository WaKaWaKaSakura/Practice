import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from d2l import torch as d2l
#读取训练集数据和测试集数据,并对数据进行预处理,
Train_Data=pd.read_csv("D:/tools/Python/pythonProject/Data/train.csv")
Test_Data=pd.read_csv("D:/tools/Python/pythonProject/Data/test.csv")
# 删除无用数据ID
Train_Data.drop('Id',axis=1,inplace=True)

#创建一个大表格里面有训练集和测试集的特征
all_features=pd.concat((Train_Data.drop('SalePrice',axis=1,inplace=False),Test_Data.drop('Id',axis=1,inplace=False)))
# 标准化
Numerical_Features=all_features.dtypes[all_features.dtypes !='object'].index
all_features[Numerical_Features]=all_features[Numerical_Features].apply(lambda x: (x-x.mean())/(x.std()))
all_features[Numerical_Features]=all_features[Numerical_Features].fillna(lambda x: x.mean())
#one-hot-code
all_features=pd.get_dummies(all_features,dtype=float)
#从all_features中读取出各个数据
N_Train=Train_Data.shape[0]
#获取训练数据和标签并转换为Tensor形式
Tensor_TrainData=torch.tensor(all_features[:N_Train].values,dtype=torch.float32)
Tensor_TrainLabels=torch.tensor(Train_Data['SalePrice'].values.reshape(-1,1),dtype=torch.float32)
#同上获取测试数据
Tensor_TestData=torch.tensor(all_features[N_Train:].values,dtype=torch.float32)

#定义一个单层的全连接网络
class Net(nn.Module):
    def __init__(self,Features_numbers):
        super(Net,self).__init__()
        self.fuction=nn.Linear(Features_numbers,1)
    def forward(self,x):
        x=self.fuction(x)
        return x
Features_number=Tensor_TrainData.shape[1]
net=Net(Features_numbers=Features_number)
#定义损失函数，这里使用均方误差。
loss=nn.MSELoss()
#定义优化器，这里使用随机梯度下降法。
optimizer=optim.SGD(net.parameters(),lr=0.003)
#取小批量数据完成训练
train_iter=d2l.load_array((Tensor_TrainData,Tensor_TrainLabels),batch_size=6)
for enpch in range(5000):
    for x,y in train_iter:
        #梯度清零
        optimizer.zero_grad()
        #向前传播得到预测值
        Prediction=net.forward(x)
        #计算损失函数
        L=loss(Prediction,y)
        #反向传播，更改参数
        L.backward()
        optimizer.step()
# #打印输出Kaggle预测
with torch.no_grad():
    #向前传播得到预测值
    Price_Pridiction=net.forward(Tensor_TestData)
    print(Price_Pridiction)
    #转化为DataFrame
    Price_Pridiction=pd.DataFrame(Price_Pridiction)
    print(Price_Pridiction)

    subssion=pd.concat((Test_Data['Id'],Price_Pridiction),axis=1)
    subssion.columns=['Id','SalePrice']
    S=subssion.to_csv(path_or_buf='D:/tools/Python/pythonProject/Data/Submission.csv',index=False)