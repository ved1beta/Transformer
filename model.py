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

        pe = torch.zeros(d_model, )
