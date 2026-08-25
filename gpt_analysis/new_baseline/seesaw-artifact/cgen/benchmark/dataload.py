import datasets
import json
import random
import numpy as np

def load_dataset(name, num_requests, tokenizer, max_length=None,
                 seed=0, dataset_path=None, input_len=None, output_len=None):
    if name == 'cnn':
        max_length = max_length or 4096
        return load_cnn(num_requests, tokenizer, max_length=max_length, seed=seed)
    elif name == 'arxiv':
        max_length = max_length or 4096 
        return load_arxiv(num_requests, tokenizer, max_length=max_length, seed=seed)
    elif name == 'sharegpt':
        max_length = max_length or 4096
        dataset_path = dataset_path or 'sharegpt.json'
        return load_sharegpt(dataset_path, num_requests, tokenizer, max_length=max_length, seed=seed)
    elif name == 'constant':
        input_len = input_len or 1024
        output_len = output_len or 256
        return constant_length(num_requests, tokenizer, input_len, output_len)
    elif name == 'test':
        return load_test(num_requests, tokenizer, seed=seed) 
    else:
        raise ValueError()

def load_test(num_requests, tokenizer, seed=0):
    random.seed(seed)
    filtered_dataset = []
    for _ in range(num_requests):
        prompt = random.choice(["The University of Toronto is located in Downtown Toronto. It is one of the top universities in the world. It has",
        "Yamanote line is a main rail line in Tokyo, Japan. It connects many important places including ",])

        prompt_len = len(tokenizer(prompt).input_ids)
        output_len = 30
        filtered_dataset.append((prompt, prompt_len, output_len))
    return filtered_dataset

def load_cnn(num_requests, tokenizer, max_length=4096, seed=0):
    random.seed(seed)
    ds = datasets.load_dataset("abisee/cnn_dailymail", "1.0.0")
    filtered_dataset = []
    for data in ds['train']:
        prompt = data['article']
        output = data['highlights']
        prompt_len = len(tokenizer(prompt).input_ids)
        output_len = len(tokenizer(output).input_ids)
        if prompt_len + output_len > max_length:
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))
        if len(filtered_dataset) >= num_requests:
            break
    return filtered_dataset

def load_arxiv(num_requests, tokenizer, max_length=8192, seed=0):
    random.seed(seed)
    ds = datasets.load_dataset("armanc/scientific_papers", "arxiv", trust_remote_code=True)
    filtered_dataset = []
    for data in ds['train']:
        prompt = data['article']
        output = data['abstract']
        prompt_len = len(tokenizer(prompt).input_ids)
        output_len = len(tokenizer(output).input_ids)
        # output_len = 1 
        if prompt_len + output_len > max_length:
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))
        if len(filtered_dataset) >= num_requests:
            break
    return filtered_dataset

def load_sharegpt(
    dataset_path: str,
    num_requests: int,
    tokenizer,
    max_length=4096,
    seed=0
):
    random.seed(seed)
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids) 
        if prompt_len < 100 or output_len < 20:
            # Prune too short sequences.
            continue
        if prompt_len + output_len > max_length:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))
    
    return filtered_dataset

def generate_prompt(
    input_len: int, tokenizer
) -> str:
    l = 0
    r = 2 * input_len
    while True:
        mid = (l + r) // 2
        tokens = np.random.randint(tokenizer.vocab_size, size=mid)
        text = tokenizer.decode(tokens)
        tokens = tokenizer.encode(text)
        if len(tokens) > input_len:
            r = mid
        elif len(tokens) < input_len:
            l = mid + 1
        else:
            return text

def constant_length(num_requests, tokenizer, input_len=1000, output_len=400):
    prompt = generate_prompt(input_len, tokenizer)
    filtered_dataset = [(prompt, input_len, output_len)] * num_requests
    return filtered_dataset