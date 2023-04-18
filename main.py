from preprocess import get_data, process_string
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import FineTuneTransformer
import torch
from timeit import default_timer as timer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = None
BOS_IDX = None
EOS_IDX = None

def padding_mask(idx_tensor):
  return (idx_tensor == PAD_IDX).transpose(0, 1)

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

        tgt_out = tgt[1:, :] # The ground truth next token. Basically the tgt_input shifted right
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

        tgt_out = tgt[1:, :] # The ground truth next token. Basically the tgt_input shifted right
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

def train(model, loss_fn, optimizer, train_loader, val_loader, n_epochs=10):
  model = model.to(device)

  for epoch in range(1, n_epochs+1):
      start_time = timer()
      train_loss = train_epoch(model, optimizer, train_loader, loss_fn)
      end_time = timer()
      val_loss = val(model, val_loader, loss_fn)
      print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

def greedy_decode(model, src, max_len, start_symbol):
    #A custom decoding method is necessary because our model's forward method is intended for teacher forcing
    src = src.to(device)
    src_padding_mask = padding_mask(src)

    memory = model.encode(src, src_padding_mask).to(device)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device) #initialize output sequence with BOS token
    for i in range(max_len-1):
        tgt_padding_mask = padding_mask(ys)
        out = model.decode(ys, memory, src_padding_mask, tgt_padding_mask).to(device) #decode into probability distribution over vocab
        out = out.transpose(0, 1).to(device) #transpose (sequence length, batch size, embedding) into (batch size, sequence length, embedding)
        prob = model.out(out[:, -1]) #get last token's probability distribution among vocab
        _, next_word = torch.max(prob, dim=1) #extract most probable token ID
        next_word = next_word.item() #extract ID from Tensor

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0) #add predicted token to our output sequence
        if next_word == EOS_IDX: #return once the end of sequence is predicted
            break
    return ys.flatten()

# function that translates from a list of tokens to another list of tokens
def translate_tokens(model, src_tokens):
    model.eval()
    src_tokens = torch.cat([torch.tensor([BOS_IDX]), src_tokens, torch.tensor([EOS_IDX])], dim=0).view(-1, 1) #tokenizer may already handle this
    num_tokens = src_tokens.shape[0]
    tgt_tokens = greedy_decode(model, src_tokens, max_len=num_tokens + 5, start_symbol=BOS_IDX)
    return tgt_tokens

# function that translates from a string sentence to a string version of the translation
def translate(model, src_sentence, tokenizer, preprocess_fn):
    model.eval()
    src_tokens = torch.tensor(preprocess_fn(src_sentence), dtype=torch.long)  
    tgt_tokens = translate_tokens(model, src_tokens)
    translation = tokenizer.decode(tgt_tokens)
    return translation

if __name__ == "__main__":
    # Initialize model and its respective tokenizer
    LLM = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    PAD_IDX = tokenizer.pad_token_id
    BOS_IDX = tokenizer.bos_token_id
    EOS_IDX = tokenizer.eos_token_id

    # Get processed data
    train_dataloader, val_dataloader, test_dataloader = get_data('JosephusCheung/GuanacoDataset')
    
    # The smaller GPT2 model we've loaded in has 50257 words in its vocabulary and uses an embedding size of 768
    embedding = LLM.get_input_embeddings()
    embed_fn = lambda x: embedding(x.long())
    model = FineTuneTransformer(LLM, tokenizer, 5, 768, 64, 50257, embed_fn, 50000, dropout=0)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    train(model, loss_fn, optimizer, train_dataloader, val_dataloader)
    val(model, val_dataloader, loss_fn)

    input_ = ""
    output = None
    while input_ != "quit":
        input_ = input("Ask a question:")
        print(translate(model, input_, tokenizer, lambda x: process_string(x, tokenizer)))
