#!/usr/bin/env python
# coding: utf-8

# In[1]:


from textblob import TextBlob
from models import InferSent
import torch
from scipy import spatial
import numpy as np


# In[2]:

def cosine_sim(x,q):
    li = []
    for item in x:
        li.append(spatial.distance.cosine(x[item],q))
    return li

def eucl_sim(x,q):
    li = []
    for item in x:
        li.append(spatial.distance.euclidean(x[item],q))
    return li


text = """ Architecturally, the school has a Catholic character. 
Atop the Main Building's gold dome is a golden statue of the Virgin Mary.
Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend 
"Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. 
Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.
It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858.
At the end of the main 
drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."""


# In[3]:
question = "What is in front of the Notre Dame Main Building?"

def calcule_cos(text,question):
    blob = TextBlob("".join(text))
    sentences = [item.raw for item in blob.sentences]

    V = 2
    MODEL_PATH = 'InferSent/encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    
    W2V_PATH = 'InferSent/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)
    infersent.build_vocab(sentences, tokenize=True)
    
    dict_embeddings = {}
    for i in range(len(sentences)):
        dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)
        encode_question = infersent.encode([question], tokenize=True)
    cos = cosine_sim(dict_embeddings,encode_question)

    return sentences,cos

def calcule_eucl(text,question):
    blob = TextBlob("".join(text))
    sentences = [item.raw for item in blob.sentences]

    V = 2
    MODEL_PATH = 'InferSent/encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    
    W2V_PATH = 'InferSent/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)
    infersent.build_vocab(sentences, tokenize=True)
    
    dict_embeddings = {}
    for i in range(len(sentences)):
        dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)
        encode_question = infersent.encode([question], tokenize=True)
    eucl = eucl_sim(dict_embeddings,encode_question)

    return sentences,eucl
"""
sentences,cos = calcule_cos(text,question)

print(sentences[np.argmin(cos)])


# In[19]:


for i in range(len(cos)):
    print("sentence ",i," : " ,sentences[i]," d'une distance de ",cos[i])


# In[ ]:

"""


