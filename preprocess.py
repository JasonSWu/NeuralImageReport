import transformers
from datasets import load_dataset
import torch
from bertopic import BERTopic

def get_data(s):
    data = load_dataset(s)
    return data

def tokenize():
    pass

def normalize():
    pass

def lowercase():
    pass

def balancing(inputs, labels=None):
    if labels is None:
        #execute balancing by topic
        pass
    else:
        #balance by label
        pass

def process_data(data, tokenizer, field):
    def tokenize_func(example):
        return tokenizer(example[field], padding="max_length", truncation=True)
    return data.map(tokenize_func, batched=True)

if __name__ == "__main__":
    print(torch.cuda.is_available())
    s = "squad"
    tokenizer = tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    data = get_data(s)
    print(data)
    question_data = process_data(data, tokenizer, 'question')
    print(data)
    answer_data = process_data(data, tokenizer, 'answers')
    print(answer_data['train']['question'][0] == question_data['train']['question'][0])
    exit()
    train_data = data['train'].train_test_split(test_size=0.2, train_size=0.8)
    val_data = train_data['test']
    train_data = train_data['train']
    test_data = data['test']