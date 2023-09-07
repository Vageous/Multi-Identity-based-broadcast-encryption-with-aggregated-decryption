from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader,TensorDataset



transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914,), (0.2023,)),
            transforms.Resize(size=224)
		])

transform_test = transforms.Compose([
			transforms.RandomCrop(32,padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914,), (0.2023,)),
            transforms.Resize(size=224)
		])

def dataset_download():
    train_set=datasets.FashionMNIST(root='./fashionmnist',train=True,download=True,transform=transform_train)
    test_set=datasets.FashionMNIST(root='./fashionmnist',train=False,download=True,transform=transform_test)
    return train_set,test_set


# if __name__=='__main__':
#     data=dataset_download()
#     train_iter=DataLoader(data[0],batch_size=2,shuffle=True)
#     for idx,batch in enumerate(train_iter):
#         print(batch[0].shape)
#         break