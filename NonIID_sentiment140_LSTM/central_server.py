import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from IBBE import IBBE

class Server(object):
    def __init__(self,args,train_set,test_set,pp) -> None:
        self.test_set=test_set
        self.layer=args.layer
        self.device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu")
        self.test_loader=DataLoader(self.test_set,batch_size=args.batchsize,shuffle=True)
        self.train_loader=DataLoader(train_set,batch_size=args.batchsize,shuffle=True)
        self.server=IBBE.server(pp)

    def model_aggregate(self,local_model,weight_model):
        for name,params in local_model.items():
            weight_model[name].add_(params)

    def cipher_model_aggregate(self,local_model,weight_model):
        
        for name,params in local_model.items():
            if name==self.layer:
                weight_model[name]=self.server.aggregate(weight_model[name],local_model[name])
            else:
                weight_model[name].add_(params)
                
    def cipher_model_recover(self,weight_model,h_prime):
        for name,params in weight_model.items():
            if name==self.layer:
                weight_model[name]=self.server.aggre_recover(h_prime,weight_model[name])
        
    def model_average(self,weight_model,num_user):
        for name,params in weight_model.items():
            weight_model[name]=weight_model[name] / num_user
        return weight_model

    def model_test(self,global_model):
        global_model.eval()
        total_loss=0.0
        correct=0
        dataset_size=0

        for batchidx,batch in enumerate(self.test_loader):
            image=batch[0].to(self.device)
            label=batch[1].to(self.device)
            dataset_size += image.size()[0]

            output=global_model(image)
            total_loss += torch.nn.functional.cross_entropy(output, label,reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc,total_l


    def model_test1(self,global_model):
        global_model.eval()
        total_loss=0.0
        correct=0
        dataset_size=0

        for batchidx,batch in enumerate(self.train_loader):
            image=batch[0].to(self.device)
            label=batch[1].to(self.device)
            dataset_size += image.size()[0]

            output=global_model(image)
            total_loss += torch.nn.functional.cross_entropy(output, label,reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc,total_l
