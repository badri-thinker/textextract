import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import nltk
import urllib
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
import sys
import re
from scipy import stats as zss
from gensim.summarization import summarize
import requests
from bs4 import BeautifulSoup
from boilerpipe.extract import Extractor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import networkx as nx
import inspect
import fastText
from textblob import TextBlob
from nltk.chunk import tree2conlltags
from nltk import ne_chunk
import spacy
from email.parser import Parser
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer
from spacy.matcher import PhraseMatcher
import sklearn_crfsuite
import fastText
import torch
from InferSent import models
TOKENIZER = RegexpTokenizer(r'\w+')
regex=re.compile('(?<=Subject:).*')
parser=Parser()


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

nlp = spacy.load('en_core_web_sm')
sq=['How much are my renewals fees for this year?',
'What is my renewal fees?',
'My renewal fees?',
'Renewal fees this year?',
'Can you tell me about my renewal fee for the year?',
'Can I know about renewal fees?',
'What is the renewal cost this year?',
'What is the cost to renew this year?',
'How much would it cost for renewal?',
'How much would i have to pay for renewal?',
'What would it cost to renew this year?',
'I would like to know what are my renewals fees for the year?',
'I am interested in knowing my renewals fee for the year?'
'What are my annual renewal fees?'
'What are my annual renewal payment?',
'How much renewal payment do I need to make this year?',
'What fees do I need to pay to renew my policy at the end of this year?']
questions=[
'How much are my renewals fees for this year?',
'What are my options for paying for my renewals?',
'How do I drop a state if I do not wish to renew it?',
'How will I be notified of my renewals?',
'Can I pay renewals online?,'
'What is the deadline to pay renewals fees?',
'What is the address where I can mail check for my renewals?,'
'How do I reverse my renewal on a state that I did not intend to renew ?',
'Can I split my renewals up into payments?',
'Can I split my renewals up using different payment methods?',
'What happens if I do not pay my renewals before the deadline?,'
'Can an RVP stop the automatic renewal debit if they do not want them debited from their earns?']

model_version=2
# modified models.py
MODEL_PATH=os.path.join("c:/USHUR/PythonProg/DATAREP/FastText","infersent2.pkl")

params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model= models.InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
W2V_PATH=os.path.join("c:/USHUR/PythonProg/DATAREP/FastText",'crawl-300d-2M.vec')
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)
X_train=np.empty((0,4096*1))
di=[]
for index, s in enumerate(questions):
    #print(type(s))
        if type(s) is not str:
            di.append(index)
            continue
        if len(s) == 0:
            di.append(index)
        #print(index)
            continue
        s=sent_tokenize(s)
        print(type(s), len(s), s)
        try:
            embeddings = model.encode(s, tokenize=True)
        except:
            di.append(index)
            continue
        X_train = np.append(X_train, embeddings.reshape(1, -1), axis=0)
print(len(di),X_train.shape)
Y_train=np.empty((0,4096*1))
di=[]
for index, s in enumerate(sq):
    #print(type(s))
        if type(s) is not str:
            di.append(index)
            continue
        if len(s) == 0:
            di.append(index)
        #print(index)
            continue
        s=sent_tokenize(s)
        print(type(s), len(s), s)
        try:
            embeddings = model.encode(s, tokenize=True)
        except:
            di.append(index)
            continue
        Y_train = np.append(Y_train, embeddings.reshape(1, -1), axis=0)
print(len(di),Y_train.shape)
for s in range(X_train.shape[0]):
    for t in range(Y_train.shape[0]):
        print(cosine(X_train[s],Y_train[t]))
exit()

for s in questions:
    doc=nlp(s)
#words=word_tokenize(s)
    chunks=[(token.text , token.pos_) for token in doc]
    print(chunks)
exit()
f=os.path.join("c:/USHUR/PythonProg/DATAREP/IL","EMAIL-TEST-DEFAULT.csv")
of=os.path.join("c:/USHUR/PythonProg/DATAREP/IL","EMAIL-TEST-DEFAULT_NVA.csv")
df_L=pd.read_csv(f,encoding='ISO-8859-1')
print(df_L.info())
df_L.dropna(inplace=True)
df_L=df_L.reset_index()
df_L.info()
print(df_L.head())
'''
df_L['topic']=df_L['topic'].astype('category')
df_L['topic']=df_L['topic'].cat.codes
'''
print(df_L.loc[0:5,'topic'])

print("+++",df_L.loc[:,'topic'].nunique())
#Y_traindf=df_L.loc[:,'topic'].reset_index(drop=True)
#print(Y_traindf.value_counts())

df_body_list=df_L['phrase']
print(len(df_body_list))

di=[]
for index, s in enumerate(df_body_list):
    if type(s) is not str:
        di.append(index)
        continue
    '''
        #s = s.split(',')
        #s = list(filter(lambda s: s != ' ', s))
        #s = list(filter(lambda s: itmatters(returntoken(s)), s))
'''    
    if len(s) == 0:
        di.append(index)
        #print(index)
        continue
    #if df_L.loc[index, 'topic'] != 'RC_Salary':
     #   continue
    sentences=sent_tokenize(s)

    emailtext=' '
    for snum, s in enumerate(sentences):
        doc = nlp(s)
        #chunks = [(chunk.root.text, chunk.root.head) for chunk in doc.noun_chunks if chunk.root.pos_ == 'NOUN']
        words=word_tokenize(s)
        chunks=[token.text for token in doc if token.pos_ == 'VERB' or token.pos_ == 'NOUN' or token.pos_=='ADJ']
        #if df_L.loc[index,'topic']=='RC_Salary':
            #print('Number of Sentences' ,len(sentences))
            #print(" words and pos",len(words),len(chunks),len(chunks)/len(words))
            #print(s)
        if chunks:
            #print(TOKENIZER.tokenize(str(chunks)))
            #if snum<=6 and len(chunks)/len(words)>0.1:
            emailtext=emailtext + ' ' + ' '.join(chunks)
        #print(chunks)
        #print("verbs",[token.text for token in doc if token.pos_=='VERB' or token.pos_=='NOUN'])

        #for ent in doc.ents:
         #   print(ent.text, ent.label_)
    #print(df_L.loc[index,'topic'],emailtext)
    df_L.loc[index,'NounPhrases']=emailtext
    #print(emailtext)
    #print("+++++++++++++++++++")
    #if index==10:
        #exit()


#print(df_L.loc[df_L.topic=='RC_Email'][['topic','NounPhrases']])

print(df_L.info())
print("empty rows",len(di))
df_L.dropna(inplace=True)
df_L.to_csv(of)
exit()

anon=os.path.join("c:/USHUR/PythonProg/DATAREP/GERDAU","gerdau_data.csv")
df_L=pd.read_csv(anon)
print(df_L.info())



for app in range(len(df_L.index)):
    body=df_L.loc[app,'REQUEST']
    s=regex.findall(body)
    if s:
        #print(s[0])
        quote=str.lower(s[0])
        doc = nlp(quote)
        chunks = [(chunk.root.text, chunk.root.head) for chunk in doc.noun_chunks if chunk.root.pos_ == 'NOUN']
        print(s[0])
        print(chunks)
        print(TOKENIZER.tokenize(str(chunks)))
        for ent in doc.ents:
            print(ent.text, ent.label_)


    #body=parser.parsestr(body)
    #print("single", body.get_payload())
    #print(body.get('From'))

exit()
#pos_nltk = nltk.pos_tag(tokenized_text)
#print("nltk tags")
#print(pos_nltk)
#print(tree2conlltags(ne_chunk(pos_nltk)))

#doc=nlp(ts)
#print("spacy ner tags")
#print([(X.text, X.label_) for X in doc.ents])

review = str.lower( "P and a pls  HOU 20 tons 18 x 42.7# 50,Hey Derek,  MC 18X42.7# GGMULTI 50'00 Rolls 6/28 at Petersburg 3pcs per bundle $61.59/cwt" )
#review="the menu looked great, and the waiter was very courteous but the food was spicy".lower()

#print(aspect_terms)
exit()

