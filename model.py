import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import copy
import math

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 0

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
def generate_subsequent_mask(size):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def padding_mask(idx_tensor):
  return (idx_tensor == PAD_IDX).transpose(0, 1)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def train_epoch(model, optimizer, train_dataloader, loss_fn):
    model.train()
    total_loss = 0

    for src, tgt in train_dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:-1, :]

        src_padding_mask = padding_mask(src)
        tgt_padding_mask = padding_mask(tgt_input)

        logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)


def val(model, val_dataloader, loss_fn):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_padding_mask = padding_mask(src)
        tgt_padding_mask = padding_mask(tgt_input)

        logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

def train(model, loss_fn, optimizer, n_epochs=10):
  model = model.to(device)

  for epoch in range(1, n_epochs+1):
      start_time = timer()
      train_loss = train_epoch(model, optimizer, train_loader, loss_fn)
      end_time = timer()
      val_loss = val(model, val_loader, loss_fn)
      print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

class ManualDecoderLayer(nn.Module):
    def __init__(self, dim_emb, dropout, nhead, dim_ff):
        super(ManualDecoderLayer, self).__init__()
        
        #TODO: Initialize the necessary pieces of the decoder block
        self.attn_weights = None
        self.residual1 = SublayerConnection(dim_emb, dropout)
        self.residual2 = SublayerConnection(dim_emb, dropout)
        self.residual3 = SublayerConnection(dim_emb, dropout)
        self.self_att = nn.MultiheadAttention(dim_emb, nhead, dropout)
        self.cross_att = nn.MultiheadAttention(dim_emb, nhead, dropout)
        self.ff = PositionwiseFeedForward(dim_emb, dim_ff, dropout)
        self.dim_emb = dim_emb

    def forward(self, x, memory_inputs, memory, src_padding_mask, tgt_mask, tgt_padding_mask):
        '''
        params:
        '''
        #TODO: Implement the forward pass
        self_att_fn = lambda input: self.self_att(input, input, input, key_padding_mask=tgt_padding_mask, attn_mask = tgt_mask)[0]
        def cross_att_fn(input):
          output, weights = self.cross_att(input, memory_inputs, memory, key_padding_mask=src_padding_mask, need_weights=True)
          self.attn_weights = weights
          return output

        output_self = self.residual1(x, self_att_fn)
        output_cross = self.residual2(output_self, cross_att_fn)
        output = self.residual3(output_cross, self.ff)

        #TODO: save the cross-attention weights here

        return output
    
class ManualDecoder(nn.Module):
    def __init__(self, layer, N):
        super(ManualDecoder, self).__init__()

        #TODO: Initialize the necessary pieces of the decoder 
        # (Hint, the mostly consists of making copies of your decoder layers)
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.dim_emb)

    def forward(self, x, memory, src_padding_mask, tgt_mask, tgt_padding_mask):

        #TODO: Implement the forward pass
        output = x
        for layer in self.layers:
          output = layer.forward(output, memory, src_padding_mask, tgt_mask, tgt_padding_mask)
        return self.norm(output)

class ManualTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, 
                 src_vocab_size, tgt_vocab_size, dim_feedforward = 512, dropout = 0.1):
        super(ManualTransformer, self).__init__()
        self.encoder = ManualEncoder(ManualEncoderLayer(emb_size, dropout, nhead, dim_feedforward), num_encoder_layers)
        self.decoder = ManualDecoder(ManualDecoderLayer(emb_size, dropout, nhead, dim_feedforward), num_decoder_layers)
        self.src_embed = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.out = nn.Linear(emb_size, tgt_vocab_size)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def get_src_mask(self, src):
        return torch.zeros((src.shape[0], src.shape[0]),device=device).type(torch.bool)

    def get_tgt_mask(self, tgt):
        return generate_subsequent_mask(tgt.shape[0])
    
    def encode(self, src, src_padding_mask):
        #TODO Implement the encode function
        output = self.encoder(self.positional_encoding(self.src_embed(src)), self.get_src_mask(src), src_padding_mask)
        return output

    def decode(self, tgt, memory, src_padding_mask, tgt_padding_mask):
        #TODO Implement the decode function
        output = self.decoder(self.positional_encoding(self.tgt_embed(tgt)), memory, src_padding_mask, self.get_tgt_mask(tgt), tgt_padding_mask)
        return output
    
    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask):
        #TODO Implement the forward pass
        encoded = self.encode(src, src_padding_mask)
        decoded = self.decode(tgt, encoded, src_padding_mask, tgt_padding_mask)
        output = self.out(decoded)
        return output