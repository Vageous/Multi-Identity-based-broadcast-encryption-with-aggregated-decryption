import torch
from torch import nn
from torchsummary import summary



class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

'''
conv.0.weight
torch.Size([96, 1, 11, 11])
conv.0.bias
torch.Size([96])
conv.3.weight
torch.Size([256, 96, 5, 5])
conv.3.bias
torch.Size([256])
conv.6.weight
torch.Size([384, 256, 3, 3])
conv.6.bias
torch.Size([384])
conv.8.weight
torch.Size([384, 384, 3, 3])
conv.8.bias
torch.Size([384])
conv.10.weight
torch.Size([256, 384, 3, 3])
conv.10.bias
torch.Size([256])
fc.0.weight
torch.Size([4096, 6400])
fc.0.bias
torch.Size([4096])
fc.3.weight
torch.Size([4096, 4096])
fc.3.bias
torch.Size([4096])
fc.6.weight
torch.Size([10, 4096])
fc.6.bias
torch.Size([10])
'''



if __name__=='__main__':
    alexnet=AlexNet()
    summary(alexnet,(1,224,224),batch_size=16,device="cpu")
#     for name,params in alexnet.state_dict().items():
#         print(name)
#         print(params.shape)