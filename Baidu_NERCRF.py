#System modules
import os
import sys

# Machine learning modules
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize

import sklearn_crfsuite
import pickle

import torch
from InferSent import models


## Helper functions -----------------------------------------------------------------------------------------------------

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# encode features for  CRF to  include sentence embedding as a feature
def doc2features(doc, i,s,vectorizer):
    word = doc[i][0]
    postag=doc[i][1]

    #Sentence embedding
    features={'s':s}

    # Features from current word
    features.update(word_vector(word,vectorizer,prefix=''))
    features.update({'postag[:2]': postag[:2]})
    # Features from previous word
    if i > 0:
        word1 = doc[i - 1][0]
        postag1 = doc[i - 1][1]
        features.update(word_vector(word1,vectorizer,prefix='w1_'))
        features.update({'-1:postag[:2]': postag1[:2]})
    else:
        features['BOS'] = True  # Special "Beginning of Sequence" tag

    # Features from next word
    if i < len(doc) - 1:
        word1 = doc[i + 1][0]
        postag1 = doc[i + 1][1]
        features.update(word_vector(word1,vectorizer,prefix='w2_'))
        features.update({'+1:postag[:2]': postag1[:2]})
        
    else:
        features['EOS'] = True  # Special "End of Sequence" tag
    return features


def extract_features(doc,vectorizer):
    return [doc2features(doc[0], i, doc[1],vectorizer) for i in range(len(doc[0]))]


def get_labels(doc):
    return [tag for (token,pos,tag) in doc]

#Define a list of expected entities, or entity related words, or typical words <stopword,name,place,object,animal,entity,property,number,etc>!!!
#Note this is suplemented by pos tag and some info learned from sentence embedding. 

WORD_LIST=[#stop words
            'the',
            'a',
            'an',
            'in',
            'for',
    
           #nouns
            'cat',
            'hat',
            'house',
            'tree',
            'bird',
            'document',
    
           #pronouns
            'He',
            'she',
            'they',
            'them',
    
           #adjective
            'hot',
            'cold',
            'new',
            'old',
            'tall',
            'short',
    
           #verb
            'running',
            'writing',
            'working',
            'talking',
            'communicating',
    
           #adverb
            'quietly',
            'quickly',
            'efficiently',
            'smoothly',
           
            #preposition
            'after',
            'before',
            'on',
            'in',
            'as',
    
            #conjunction
            'next',
            'also',
            'secondly',
            'eventually',
    
            #Numeric 
            '23224',
            '423',
            '4',
    
            #Entities appearing in training phrases.
            'Andrew',
            'Ushur',
            'Northwestern',
            'Portland',
            'group',
            'life',
            'DI',
            '10',
            'employees',
            'equity'
            ]

def word_vector(word,vectorizer,prefix=''):
    word=word.lower()
    features={}
    for x in WORD_LIST:
        x=x.lower()
        val=cosine(vectorizer(word),vectorizer(x))
        if val>0.6:
            features[prefix+x]=val*10
        else:
            features[prefix+x]=0
    return features
## Load the pretrained fasttext word embeddings and pretrained Infersent model ------------------------------------------

# modified models.py to overcome stride issue

MODEL_PATH=os.path.join("c:/USHUR/PythonProg/DATAREP/FastText","infersent2.pkl")
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
model= models.InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

#W2V_PATH=os.path.join("dataset/fastText","crawl-300d-2M-subword.vec")
#model.set_w2v_path(W2V_PATH)

# Load fastText model from a bin file (subseqently used for word encoding)
FAST_TEXT_MODEL_PATH=os.path.join("c:/USHUR/PythonProg/DATAREP/FastText","crawl-300d-2M-subword.bin")
model.load_fastText_model(FAST_TEXT_MODEL_PATH)

# Get the fasttext word embedding function.
vectorizer = model.word_vec


## Create and format the training data ----------------------------------------------------------------------------------
# labelled data to train CRF from custom NER tags
data=[(['I', 'work', 'with', 'Andrew', 'Usher', 'here', 'at', 'Northwestern', 'and', 'we', 'have', 'a', 'new', 'company', 'here', 'in', 'Portland', 'on', 'both', 'a', 'group', 'life', 'and', 'group', 'DI', 'plan', '.', 'As', 'you', 'can', 'see', 'from', 'the', 'attached', 'census', 'there', 'are', 'currently', '10', 'employees', ',', 'all', 'of', 'which', 'have', 'equity', 'in', 'the', 'company', '.', 'This', 'company', 'currently', 'does', 'not', 'have', 'a', 'plan', 'in', 'place', 'for', 'life', 'or', 'DI', '.', 'Ignore', 'the', 'info', 'below', 'pertaining', 'to', 'the', 'health', 'insurance', '.'],
       
      ['IR', 'IR', 'IR', 'V-B-BROKER', 'V-I-BROKER', 'IR', 'IR', 'V-B-COMPANY', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'V-B-LOC', 'IR', 'IR', 'IR', 'V-B-PLAN', 'V-I-PLAN', 'IR', 'V-B-PLAN', 'V-I-PLAN', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'G-B-NUM', 'N-B-EMP', 'IR', 'IR', 'IR', 'IR', 'IR', 'V-EMPPROP', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'V-I-PLAN', 'IR', 'V-I-PLAN', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR', 'IR'])]

# create a corpus  that includes a  sentence embedding converted to a string
# CRF does not take vectors of real numbers for feature set
# converted 4096 embedding to a string of p and n (p for positive, n for negative)
# convert  real numbers to  discrete representation string of binary chars (p,n)
# there are other methods to convert embedding to discrete features; we can try them later

corpus = []
for (doc, tags) in data:
    postags=nltk.pos_tag(doc)
    doc_tag = []
    i=0
    for word, tag in zip(doc,tags):
        doc_tag.append((word,postags[i][1],tag))
        i=i+1
    s=" ".join(doc)
    z = np.where(model.encode([" ".join(doc)]) >= 0, 'p', 'n').astype('|S1').tostring().decode('utf-8')

    corpus.append((doc_tag,z))
    
X = [extract_features(doc,vectorizer) for doc in corpus]
y = [get_labels(doc[0]) for doc in corpus]

## Train the CRF model ---------------------------------------------------------------------------------------------------

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

fcrf=crf.fit(X,y)

## Make predictions using the trained CRF model ------------------------------------------------------------------

ts='Hi my name is John Wood. I work at MobiLife.  Our new company here in Austin offers life and DI plans . We currently have 50 employees, with universal equity in the company. We do not yet have a plan for life or DI. Ignore the info below - it is automatically generated, and not of concern.'
s=model.encode(ts)
z=np.where(s>np.mean(s)+np.std(s),'p',(np.where(s<np.mean(s)-np.std(s),'n','0'))).astype('|S1').tostring().decode('utf-8')
test = (nltk.pos_tag(word_tokenize(ts)),z)

X_test = extract_features(test,vectorizer)
ans=fcrf.predict_single(X_test)
print("NER tags recognized by CRF:")
print(ans)
