import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import copy
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
class ManualDecoder(nn.Module):
    def __init__(self, layer, N):
        super(ManualDecoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.dim_emb)

    def forward(self, x, encoded, memory, memory_masks, src_padding_mask, tgt_mask, tgt_padding_mask, training):
        output = x
        if not training:
            for mem, mask in zip(memory, memory_masks):
                output = self.layers[0].forward(output, mem, mask, tgt_mask, tgt_padding_mask)
        for layer in self.layers:
          output = layer.forward(output, encoded, src_padding_mask, tgt_mask, tgt_padding_mask)
        return self.norm(output)

class FineTuneTransformer(nn.Module):
    def __init__(self, LLM, tokenizer, num_decoder_layers, emb_size, nhead, tgt_vocab_size, embed_fn, dim_feedforward = 512, dropout = 0.1):
        super(FineTuneTransformer, self).__init__()
        self.encoder = LLM
        self.decoder = ManualDecoder(nn.TransformerDecoderLayer(emb_size, nhead, dim_feedforward, dropout), num_decoder_layers)
        self.tokenizer = tokenizer
        self.embed = embed_fn
        self.out = nn.Linear(emb_size, tgt_vocab_size)
        self.memory = []
        self.memory_masks = []
    
    def get_src_mask(self, src):
        # Essentially no masking
        return torch.zeros((src.shape[0], src.shape[0]),device=device).type(torch.bool)

    def get_tgt_mask(self, tgt):
        # For use in training while teacher forcing. Do not want tokens to cheat and look at the future.
        # Generates triangular mask
        size = tgt.shape[0]
        mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def encode(self, src, src_padding_mask, token_type_ids):
        output = self.encoder(input_ids=src, attention_mask=src_padding_mask, token_type_ids=token_type_ids)
        return output

    def decode(self, tgt, encoded, src_padding_mask, tgt_padding_mask):
        output = self.decoder(
            self.embed(self.tokenizer(tgt)), encoded, self.memory, self.memory_masks, 
            src_padding_mask, self.get_tgt_mask(tgt), tgt_padding_mask, self.training)
        return output
    
    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask):
        # Note that this implementation is for teacher forcing.
        encoded = self.encode(src, src_padding_mask)
        if not self.training:
            self.memory.append(encoded)
            self.memory_masks.append(src_padding_mask)
        decoded = self.decode(tgt, encoded, src_padding_mask, tgt_padding_mask)
        output = self.out(decoded)
        return output