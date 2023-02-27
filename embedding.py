import openai, json, os
import numpy as np
openai.api_key = os.environ["OPEN_AI_KEY"]
#ambiguous sentences = I saw her duck; The chicken is ready to eat

def Embedding(content, engine='text-similarity-ada-001'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']
    return vector

def similarity(v1,v2):
    return np.dot(v1, v2)
    
if __name__ == '__main__':

    your_word = input('Enter a term: ')
    your_embedded_word = Embedding(your_word)
    
    #print(your_embedded_word)
    #exit()

    categories = ['plant', 'reptiles', 'mammals', 'fish', 'primates', 'birds']
    #categories = ['animal', 'action']
    scores = list()
    for category in categories:
        scores.append({'category':category, 'score':similarity(your_embedded_word,Embedding('type of ' + category))})

    scores = sorted(scores, key=lambda x: x['score'], reverse=True)  

    print(json.dumps(scores, indent=4) + '\n')
    
    #print(f"\"{your_word.upper()}\" typically falls under the category of \"{scores[0]['category'].upper()}\"")
