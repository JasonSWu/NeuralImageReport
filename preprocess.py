import transformers
from datasets import load_dataset
import torch

def get_data(s):
    data = load_dataset(s)
    return data

def process_data(data):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    def tokenize_func(example):
        return tokenizer(example['text'], padding="max_length", truncation=True)
    data.map(tokenize_func, batched=True)

if __name__ == "__main__":
    print(torch.cuda.is_available())
    s = "imdb"
    data = get_data(s)
    process_data(data)
    print(data)