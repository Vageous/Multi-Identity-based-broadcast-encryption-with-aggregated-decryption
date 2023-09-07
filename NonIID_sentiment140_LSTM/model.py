import torch
from config import parser_args
from torchsummary import summary
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.args=parser_args()
        self.device=torch.device("cuda" if torch.cuda.is_available()==self.args.device else "cpu")
        # 1. Feed the tweets in the embedding layer
        # padding_idx set to not learn the emedding for the  token - irrelevant to determining sentiment
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.embedding_dim)

        # 2. LSTM layer
        # returns the output and a tuple of the final hidden state and final cell state
        self.lstm = nn.LSTM(self.args.embedding_dim, 
                               self.args.hidden_dim, 
                               num_layers=self.args.num_layers,
                               bidirectional=self.args.bidirectional,
                               dropout=self.args.dropout)
                            #   batch_first=True)
        
        # 3. Fully-connected layer
        # Final hidden state has both a forward and a backward component concatenated together
        # The size of the input to the nn.Linear layer is twice that of the hidden dimension size
        if self.args.bidirectional:
            self.fc = nn.Linear(self.args.hidden_dim*2, self.args.output_dim)
        else:
            self.fc = nn.Linear(self.args.hidden_dim, self.args.output_dim)

        # Initialize dropout layer for regularifc
        self.dropout = nn.Dropout(self.args.dropout)
      
    def forward(self, x):
        batch_size, seq_len = x.shape
        #初始化一个h0,也即c0，在RNN中一个Cell输出的ht和Ct是相同的，而LSTM的一个cell输出的ht和Ct是不同的
        #维度[layers, batch, hidden_len]
        if self.args.bidirectional:
            h0 = torch.randn(self.args.num_layers*2, batch_size, self.args.hidden_dim).to(self.device)
            c0 = torch.randn(self.args.num_layers*2, batch_size, self.args.hidden_dim).to(self.device)
        else:
            h0 = torch.randn(self.args.num_layers, batch_size, self.args.hidden_dim).to(self.device)
            c0 = torch.randn(self.args.num_layers, batch_size, self.args.hidden_dim).to(self.device)

        x = self.dropout(self.embedding(x))
        out,(_,_)= self.lstm(x, (h0,c0))
        output = self.fc(out[:,-1,:]).squeeze(0)
        return output
# class LSTM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.args=parser_args()

#         self.embedding = nn.Embedding(self.args.vocab_size, self.args.embedding_dim)
#         self.lstm = nn.LSTM(self.args.embedding_dim, self.args.hidden_dim, num_layers=self.args.n_layers, bidirectional=self.args.bidirectional, dropout=self.args.dropout)
#         self.fc = nn.Linear(self.args.hidden_dim * 2 if self.args.bidirectional else self.args.hidden_dim, self.args.output_dim)
#         self.dropout = nn.Dropout(self.args.dropout)
    
#     def forward(self, text):
#         embedded = self.dropout(self.embedding(text))
#         output, (hidden, cell) = self.lstm(embedded)
#         if self.lstm.bidirectional:
#             hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
#         else:
#             hidden = self.dropout(hidden[-1,:,:])
#         out = self.fc(hidden)
#         return out


  

if __name__=='__main__':
    model=LSTM()
    for name,params in model.state_dict().items():
        print(name)
        print(params.size())

