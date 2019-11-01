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
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import stop_words
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

#print(stopwords.words('english'))
#print(stop_words.ENGLISH_STOP_WORDS)
exit()
TOKENIZER = RegexpTokenizer(r'\w+')
regex=re.compile('(?<=Subject:).*')
parser=Parser()
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

nlp = spacy.load('en_core_web_sm')
#f=os.path.join("c:/USHUR/PythonProg/DATAREP/IL","EMAIL-TRAIN.csv")
f=os.path.join("c:/USHUR/PythonProg/DATAREP/IL","EMAIL-NV-TRAIN-CSV.csv")
df_L=pd.read_csv(f)
print(df_L.info())
df_L.dropna(inplace=True)
df_L=df_L.reset_index()
df_L.info()
print(df_L.head())

#df_L['topic']=df_L['topic'].astype('category')
#df_L['topic']=df_L['topic'].cat.codes
print(df_L.loc[0:5,'topic'])
#print("+++",df_L.loc[:,'topic'].nunique())
#Y_traindf=df_L.loc[:,'topic'].reset_index(drop=True)
#print(Y_traindf.value_counts())

s1=word_tokenize('The information contained in this message may be privileged and confidential and protected from disclosure. ' \
'If the reader of this message is not the intended recipient, or an employee or agent' \
'responsible for delivering this message to the intended recipient, you are hereby notified' \
'that any dissemination, distribution or copying of this communication is strictly prohibited. If you have received this communication in error, ' \
'please notify us immediately by replying to the message and deleting it from your computer. Notice required by law: This e-mail may ' \
'constitute an advertisement or solicitation under U.S. law, if its primary purpose is to advertise or promote a commercial ' \
'product or service. You may choose not to receive advertising and promotional messages from Ernst & Young LLP ' \
'(except for EY Client Portal and the ey.com website, which track e-mail preferences through a separate process) at this e-mail ' \
'address by forwarding this message to no-more-mail@ey.com. If you do so, the sender of this message will be notified promptly. ' \
'Our principal postal address is 5 Times Square, New York, NY 10036. Thank you. Ernst & Young LLP')

'This e - mail message is confidential and may contain legally privileged information . If you are not the ' \
'intended recipient you should not read , copy , distribute , disclose or otherwise use the information in ' \
'this e - mail . Please also telephone us on : + USHUR_NUMBER ( USHUR_NUMBER USHUR_NUMBER or fax us on + USHUR_NUMBER ( USHUR_NUMBER USHUR_NUMBER immediately ' \
'and delete the message from your system . E - mail may be susceptible to data corruption , interception and ' \
'unauthorised amendment , and we do not accept liability for any such corruption , interception or amendment or the consequences thereof . ' \
'Although we have taken reasonable precautions to ensure that no viruses are present in this email , we can not accept responsibility for any loss or damage arising from the use of this ' \
'email or its attachments.'


cvect=CountVectorizer(stop_words='english')


vect = TfidfVectorizer(stop_words='english', max_df=0.50, min_df=2)
X = vect.fit_transform(df_L.NounPhrases)
X_dense = X.todense()
coords = PCA(n_components=2).fit_transform(X_dense)
plt.scatter(coords[:, 0], coords[:, 1], c='m')
plt.show()

def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df
def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)
features = vect.get_feature_names()
print(top_feats_in_doc(X, features, 1, 10))
n_clusters = 16
clf = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=1)
labels = clf.fit_predict(X)
print(labels.shape)

print("lables" ,labels)

#plt.scatter(X[:, 0], X[:, 1],c='m')

centers = clf.cluster_centers_
print(centers.shape)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
exit()
df_body_list=df_L['phrase_processed']
print(len(df_body_list))

di=[]
for index, s in enumerate(df_body_list):
    if type(s) is not str:
        di.append(index)
        continue
    '''
        s = s.split(',')
        s = list(filter(lambda s: s != ' ', s))
        s = list(filter(lambda s: itmatters(returntoken(s)), s))
    '''
    if len(s) == 0:
        di.append(index)
        #print(index)
        continue

    sentences=sent_tokenize(s)
    print(len(sentences))
    emailtext=' '
    for s in sentences:
        doc = nlp(s)
        #chunks = [(chunk.root.text, chunk.root.head) for chunk in doc.noun_chunks if chunk.root.pos_ == 'NOUN']
        chunks=[token.text for token in doc if token.pos_ == 'VERB' or token.pos_ == 'NOUN']
        if chunks:
            #print(TOKENIZER.tokenize(str(chunks)))
            emailtext=emailtext + ' ' + ' '.join(chunks)
        #print(chunks)
        #print("verbs",[token.text for token in doc if token.pos_=='VERB' or token.pos_=='NOUN'])

        #for ent in doc.ents:
         #   print(ent.text, ent.label_)
    #print(df_L.loc[index,'topic'],emailtext)
    df_L.loc[index,'NounPhrases']=emailtext
    #print(emailtext)
    #print("+++++++++++++++++++")


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

