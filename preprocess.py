import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import torch
from torch.utils import data
from bertopic import BERTopic

def process_string(s, tokenizer):
    # Process a single string into a tensor of IDs
    input_ids = tokenizer.encode(s, add_special_tokens=True, return_tensors='pt')
    return input_ids

def tokenize(data_, tokenizer):
    def tokenize_func(example):
        return tokenizer(example['question'], padding="max_length", truncation=True)
    return data_.map(tokenize_func, batched=True)

def normalize(data_):
    # Normalize the text data by removing any special characters, etc.
    pass

def lowercase(data_):
    # Convert all text to lowercase
    pass

def balance(data_):
    pass

def clean(data_):
    pass

def preprocess(data_, tokenizer):
    # Prerocess a dataset
    data_ = clean(data_)
    data_ = lowercase(data_)
    data_ = tokenize(data_, tokenizer)
    data_ = normalize(data_)
    return data_

def get_data(s):
    data_ = load_dataset(s)
    data_ = preprocess(data_)
    my_dataset = MyDataset(data_['train'])
    dataloader = data.Dataset(my_dataset, batch_size=32, shuffle=True)
    return dataloader

# define a custom PyTorch dataset class
class MyDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.dataset[idx]['input_ids'],
            'attention_mask': self.dataset[idx]['attention_mask'],
            'labels': self.dataset[idx]['label']
        }

if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    embedding = model.get_input_embeddings()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    embedding(torch.tensor([1,2,3], dtype=torch.long))
    print(tokenizer.encode("what's up, how is ya?", add_special_tokens=False))
    exit()
    print(torch.cuda.is_available())
    s = "squad"
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    data = get_data(s)
    print(data['train']['answers'][0])
    question_data = process_data(data, tokenizer)
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    answer_data = process_data(data, tokenizer, 'answers')
    print(answer_data['train']['question'][0] == question_data['train']['question'][0])
    train_data = data['train'].train_test_split(test_size=0.2, train_size=0.8)
    val_data = train_data['test']
    train_data = train_data['train']
    test_data = data['test']