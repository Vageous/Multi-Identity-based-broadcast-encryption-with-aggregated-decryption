import argparse

def parser_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epoch",default=40)
    parser.add_argument("--batchsize",default=16)
    parser.add_argument("--lr",default=0.001)
    parser.add_argument("--num_user",default=50)
    parser.add_argument("--local_round",default=1)
    parser.add_argument("--frac",default=1)
    parser.add_argument("--bits",default=20)
    parser.add_argument("--ID",default=["user1","user2"])
    parser.add_argument("--scale",default=10000)
    parser.add_argument("--flag",default=0)
    parser.add_argument("--layer",default="conv2.weight")
    parser.add_argument("--device",default=1)
    parser.add_argument("--rate",default=0.3)
    parser.add_argument("--droptype",default=0)
    parser.add_argument("--filepath",type=list,default=['train_text.pt','train_label.pt','test_text.pt','test_label.pt'])
    # LSTM 
    # vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout
    parser.add_argument("--vocab_size",type=int,default=100000)
    parser.add_argument("--embedding_dim",type=int,default=100)
    parser.add_argument("--hidden_dim",type=int,default=256)
    parser.add_argument("--output_dim",type=int,default=2)
    parser.add_argument("--num_layers",type=int,default=4)
    parser.add_argument("--bidirectional",default=True)
    parser.add_argument("--dropout",default=0.3)

    args=parser.parse_args()
    return args
    
