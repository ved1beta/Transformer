import torch 
import torch.nn as nn
import math
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size:int):
        super.__init__()
        self.d_model = d_model
        self.vocab_size= vocab_size
        self.embeddings = nn.Embedding(d_model, vocab_size)
    
    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)
        
class PositionalEncoding(nn.Model):
    def __init__(self, d_model:int, seq_length:int,dropout:float ) -> None:
        super.__init__()
        self.d_model =d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_length, d_model )

        position = torch.arange(0 , seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # sin and cos sepration : ) 
        pe[:, 0::2]= torch.sin(position* div_term)
        pe[:, 1::2]= torch.cos(position* div_term)

        pe = pe.unsqueeze(0) # stores horizontaly 

        self.register_buffer("pe", pe)


    def forward(self , x):
        x = x + (self.pe[: , : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __inti__(self, esp: float = 10**-6)->None: 
        super.__init__()
        self.esp= esp
        self.alpha = nn.Parameter(torch.ones(1)) # x using one note recomm 
        self.bias = nn.Parameter(torch.zeros(1)) # + using 1 not recom : ) 

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x-mean)/ (std + self.esp) + self.bias
