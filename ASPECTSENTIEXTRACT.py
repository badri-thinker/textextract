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
from nltk.tokenize import word_tokenize
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

