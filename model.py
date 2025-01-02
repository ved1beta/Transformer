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
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int , d_ff:int ,  dropout:float)-> float:
        super().__init__()
        # (xW1 + B)W2+ B2
        self.liner1 = nn.Linear(d_model, d_ff) # 
        self.dropout = nn.Dropout(dropout)
        self.liner2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.liner2(self.dropout(torch.relu(self.liner1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int , dropout:float)-> None:
        super().__init__()
        self.d_model = d_model
        self.h = h 
        assert d_model % h == 0 

        self.d_k = d_model/h  

        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(query , key , value , mask , dropout: nn.Dropout):
        d_k = query.shape[1]

        attention_score = (query @ key.transpose(-2, -1) )/math.sqrt(d_k)

        if mask is not None:
            attention_score.masked_fill_(mask == 0 , -1e9)
        attention_score = attention_score.softmax( dim = -1)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return(attention_score @ value), attention_score
    

    def forward(self , x, k, q, v ,mask ):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h , self.d_k).transpose(1,2)

        key = query.view(key.shape[0], key.shape[1], self.h , self.d_k).transpose(1,2)

        value = query.view(value.shape[0], value.shape[1], self.h , self.d_k).transpose(1,2)

        x, self.attention = MultiHeadAttentionBlock.attention(query, key , value , mask , self.dropout)

        x= x.transpose(1, 2).contiguous().view(x.shape[0], -1 , self.h * self.d_k )

        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None :
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
        
class EncoderBlock(nn.Module):
    def __init__(self, selfattention_block : MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock , dropout : float)-> None:
        super().__init__()
        self.self_attention_block = selfattention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.Module([ResidualConnection(dropout)for _ in range(2)])

    def forward(self, x , src_mask ):
        x = self.residual_connection[0](x, lambda x : self.self_attention_block(x, x, x,src_mask))
        x= self.residual_connection[1](x, self.feed_forward_block)

        return x
class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:

        super().__init__()
        self.norm = LayerNormalization()

    def forward(self, x, mask ):
        for layer in self.layers:
            x= layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connections = nn.Module([ResidualConnection(dropout)for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask ):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x, tgt_mask ))
        x= self.residual_connections[1](x, lambda x:self.cross_attention_block(x,encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x 
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):    

    def __init__(self , encoder :Encoder, decoder :Decoder, src_embed :InputEmbeddings, tgt_embed : InputEmbeddings, src_pos: PositionalEncoding, tgt_pos:PositionalEncoding , projection_layer : ProjectionLayer):

        super().__init__()
        self.encoder  = encoder
        self.decoder = decoder
        self.src_emded = src_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.tgt_embed = tgt_embed
        self.projection_laye = projection_layer

    def encode(self, src, src_mask):
        src = self.src_emded(src)
        src= self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt,  tgt_mask):
        tgt= self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x ):
        return self.projection_laye(x)

def build_transformer(src_vocab_size : int , tgt_vocab_size :int,src_seq_len :int, tgt_seq_len :int , d_model:int =512,N:int = 6, h:int=8, dropout:float= 0.1,  d_ff:int= 2048)-> Transformer:

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model , tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff , dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range (N):
        decoder_self_attention_block= MultiHeadAttentionBlock(d_model , h, dropout)
        decoder_cross_attention_block= MultiHeadAttentionBlock(d_model , h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blcok = DecoderBlock(decoder_cross_attention_block, decoder_self_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_blcok)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blcok))

    projectionLayer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder , src_embed, tgt_embed, src_pos, tgt_pos,projectionLayer)

    # initializing paramaters 
    for p in transformer.parameters():
        if p.dim() > 1 :
            nn.init.xavier_uniform_(p)

    return transformer 






    
 
