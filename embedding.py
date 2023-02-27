import openai, json, os
import numpy as np
openai.api_key = os.environ["OPEN_AI_KEY"]

def gpt3_embedding(content, engine='text-similarity-ada-001'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']
    return vector

def similarity(v1,v2):
    return np.dot(v1, v2)
    
if __name__ == '__main__':

    your_word = input('Enter a life form: ')
    your_embedded_word = gpt3_embedding(your_word)
    
    print(your_embedded_word)
    exit()

    categories = ['plant', 'reptiles', 'mammals', 'fish', 'primates']
    scores = list()
    for category in categories:
        scores.append({'category':category, 'score':similarity(your_embedded_word,gpt3_embedding('type of ' + category))})

    scores = sorted(list_input, key=lambda x: x['score'], reverse=True) 

    print(json.dumps(scores, indent=4))
