import transformers

def tokenize(prompt): 
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai-gpt")
    tokens = tokenizer.tokenize(prompt)
    return tokens
    
string = input('Enter a term to tokenize: ')

print(tokenize(string))