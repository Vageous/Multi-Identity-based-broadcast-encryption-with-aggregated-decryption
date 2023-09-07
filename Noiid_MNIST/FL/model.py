# cnn model
'''
conv1.weight: torch.size([20,1,5,5])
conv1.bias: torch.size([20])
conv2.weight: torch.size([40,20,5,5])
conv2.bias: torch.size([40])
fc1.weight: torch.size([120,1440])
fc1.bias: torch.size([120])
fc2.weight: torch.size([84,120])
fc2.bias: torch.size([84])
fc3.weight: torch.size([10,84])
fc3.bias: torch.size([10])
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torchsummary import summary

class cnn(nn.Module):
    def __init__(self) -> None:
        super(cnn,self).__init__()
        self.conv1=nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0)
        self.fc1=nn.Linear(16*6*6,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        # conv2 output_channel*num*num*batchsize
        x=x.view(-1,16*6*6)

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x
    
class cnn1(nn.Module):
    def __init__(self) -> None:
        super(cnn1,self).__init__()
        self.conv1=nn.Conv2d(3,6,kernel_size=5,stride=1,padding=2)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        # conv2 output_channel*num*num*batchsize
        x=x.view(-1,16*5*5)

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x
# test=cnn()
# print(test)
def Net():
    num_classes = 10
    model = resnet18(pretrained=True)
    n = model.fc.in_features
    model.fc = nn.Linear(n, num_classes)
    return model


if __name__=='__main__':
    # test1=cnn1()
    # summary(test1,(3,28,28),32,'cpu')
    net = Net()
    summary(net.cuda(), (3, 32, 32))