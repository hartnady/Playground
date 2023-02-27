import transformers, tiktoken

def tokenize_words(prompt):  
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai-gpt")
    tokens = tokenizer.tokenize(prompt)
    return str(tokens)
    
def tokenize_semantic(string, encoding_name='cl100k_base'): 
    encoding = tiktoken.get_encoding(encoding_name)     
    return str(encoding.encode(string))
    
string = input('Enter a term to tokenize: ')

print('Word Tokens: ' + tokenize_words(string))
print('Numeric Tokens: ' + tokenize_semantic(string))