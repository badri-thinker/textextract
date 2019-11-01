import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as plt
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
# from boilerpipe.extract import Extractor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# import matplotlib.pyplot as plt
# import seaborn as sns
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import networkx as nx
import inspect
# import fastText
from textblob import TextBlob
from nltk.chunk import tree2conlltags
from nltk import ne_chunk
import spacy
from email.parser import Parser
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer
from spacy.matcher import PhraseMatcher
import sklearn_crfsuite
# import fastText
import torch
from InferSent import models
from sklearn.model_selection import train_test_split
from keras import Sequential, initializers, regularizers, constraints
from keras.backend import squeeze,dot,expand_dims,tanh,sum, cast, epsilon,floatx, exp
from keras.layers import Dense, Input,   Reshape, Conv2D, MaxPool2D, concatenate
from keras.models import Model
from keras.layers import Layer,Embedding, Flatten, Softmax,SpatialDropout1D, LSTM, Bidirectional,GRU, Dropout, Activation
from keras.layers import MaxPooling1D, AveragePooling1D,Conv1D,Concatenate, GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import optimizers
from keras.utils import to_categorical
from rank_bm25 import BM25Okapi
from rake_nltk import rake
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
r=rake.Rake()
summary='The Smart Home is here and homeowners are going to be overwhelmed with connected devices. ' \
        ' Axius simplifies the entire process of managing a Smart Home from delivery to security to ' \
        'support. We distribute our service via Builders, Integrators, and Property Managers who ' \
        'benefit from a dedicated partner and Smart Home expert to assist with project implementation ' \
        'and homeowner support. Axiusâ€™ customized services save time and money while increasing ' \
        'homeowner satisfaction. Our unique combination of hardware, software, and best-in-class IT support ' \
        'offers a personalized, remote service solution for smart homeowners across all \
     of their connected devices. Homeowners enjoy the value technology brings to their homes with' \
        ' Axius as their single point of contact to handle any questions or complexities. ' \
        'The result is a thriving Smart Home with the best products selected, all devices connected, a' \
        'nd the network protected.  Learn more about us at https://www.axi.us.'
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer('[A-Za-z]\w+')
#word_tokens = word_tokenize(summary)
#filtered_sentence = [w for w in word_tokens if not w in stop_words]
#print(filtered_sentence)
#listToStr = ' '.join(map(str, filtered_sentence))
#print(listToStr)
fsummary=os.path.join("c:/USHUR/PythonProg/DATAREP/PNP","PNP-Insurtech.csv")
fkeywords = os.path.join("c:/USHUR/PythonProg/DATAREP/PNP",'PNP-KEYWORDS.csv')
df_L=pd.read_csv(fkeywords)

TC=TfidfVectorizer(max_features=100,stop_words=stop_words,ngram_range=(1,1),tokenizer=tokenizer.tokenize)
def extract_kw(summary):
    r.extract_keywords_from_text(summary)
    rkw=[kw for kw in r.get_ranked_phrases() if len(kw.split())==1]
    #print("RAKE",rkw)
    X=TC.fit_transform(sent_tokenize(summary))
    indices = np.argsort(TC.idf_)
    tkw=TC.get_feature_names()
    top_n = 10
    top_keywords = [tkw[i] for i in indices[:top_n][::-1]]
    #print("TFIDF",top_keywords)
    best_keywords=[x for x in set(rkw) if x in set(top_keywords)]
    #print("BestKW",best_keywords)
    #print("differencerake", set(rkw).difference(set(top_keywords)))
    extrakw=list(set(top_keywords).difference(set(rkw)))
    if len(extrakw) >=3:
        best_keywords.extend(extrakw[:3])
    print("best",best_keywords)
    return best_keywords


df_L['UNI-KEYWORDS']=df_L['Description'].apply(lambda x:extract_kw(x))


df_L.to_csv(fkeywords)
exit()
X=X.toarray()
# determine  a tfidf matrix  for the entire corpus:  trigrams
topn_tri=30
print("number of training trigrams",topn_tri)
for app  in range(len(XY_train.index)):
    #find the indices for this app with the highest tfidf scores
    topn_ids = np.argsort(X[app])[::-1][:topn_tri].tolist()
exit()

corpus = ['better job States',
'Can I speak my when you are up to in two payments?',
'Can I split my payment into two parts?',
'Can I split my payment up to in two payments?',
'Can I stop the automatic renewal?',
'Connect if I different payment my thirst.',
'Who do I make my check out too.',
'f******* my payment method',
'How do I get married?'
'Hey, I forgot to send my check mark should I do?',
'Hey, I forgot to send my check. What should I do?',
'Hey, what are my options for paying online?',
'Hi comma stop my auto payment.',
'is there life on mars?',
'Hi, how do I drop the state if I do not wish to win?',
'Hi, Mike, Damon Mazurek.',
'How do I pay with something other than the card you have for me?',
'How we like it notification?',
'How you do not intend to renew? What should I do?',
'I like to pay by check I can I do too.',
'ladies Insurance necessary',
'How do i drive a truck?',
'Okay drop estate.',
'paying by check',
'Venom a payment due',
'How do I  pay for chicken?',
'How do I cook a chicken?',
'can I make a pie?',
'How can I renew my life',
'Are there any animals there',
'What is the plan for the new?',
'When is there an hour period?',
'Where do I know my truck too?',
'Why do I need insurance?',
]
ftrain=os.path.join("c:/USHUR/PythonProg/DATAREP/PRIMERICA","QTRAIN.csv")
df_L=pd.read_csv(ftrain)
df_L.info()
df_L.dropna(inplace=True)
df_L.info()
print(df_L.head())
df_L['topic']=df_L['topic'].astype('category')
print(df_L.loc[0:5,'topic'])
topic_d = dict(enumerate(df_L['topic'].cat.categories))
print("dictionary",topic_d, topic_d[2])

df_L['topic_ID']=df_L['topic'].cat.codes
df_body_list=df_L['question']
tokenized_corpus = [doc.split(" ") for doc in df_body_list]
print(tokenized_corpus)
bm25 = BM25Okapi(tokenized_corpus)
query = "Cathy my renewal but the credit card on a check"
tokenized_query = query.split(" ")
doc_scores = bm25.get_scores(tokenized_query)
print(np.argmax(doc_scores),np.amax(doc_scores),doc_scores.argsort()[-3:][::-1],doc_scores[doc_scores.argsort()[-3:][::-1]])
print(df_L.loc[np.argmax(doc_scores),['question','topic']])
print(bm25.get_top_n(query,tokenized_corpus,1))
print("docsocres",doc_scores)
exit()
# array([0.        , 0.93729472, 0.        ])
TOKENIZER = RegexpTokenizer(r'\w+')
regex=re.compile('(?<=Subject:).*')
parser=Parser()

ftrain=os.path.join("c:/USHUR/PythonProg/DATAREP/PRIMERICA","QTRAIN.csv")
df_L=pd.read_csv(ftrain)
df_L.info()
df_L.dropna(inplace=True)
df_L.info()
print(df_L.head())
df_L['topic']=df_L['topic'].astype('category')
print(df_L.loc[0:5,'topic'])
topic_d = dict(enumerate(df_L['topic'].cat.categories))
print("dictionary",topic_d, topic_d[2])

df_L['topic_ID']=df_L['topic'].cat.codes
#df_R=pd.read_csv(rfr)
#df_R['TOPIC_NAME']=df_R['label'].map(d)
#df_R.to_csv(ofrw)

#df_L['TOPIC_NAME']=df_L['topic_ID'].map(d)
Topic_names=pd.Series(topic_d)
Topic_names=Topic_names.values.reshape(-1,1)
print(type(Topic_names),Topic_names.shape)
print(Topic_names[:,0])


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


def embedtext(xtrain, df_body_list):
    di=[]
    for index, s in enumerate(df_body_list):
        #print(type(s),s)
        if type(s) is not str:
            di.append(index)
            continue
        if len(s) == 0:
            di.append(index)
            #print(index)
            continue
        s=sent_tokenize(s)
        #print(type(s), len(s), s)
        try:
            qembedding = model.encode(s, tokenize=True)
            #print(type(qembedding),qembedding.shape)
        except:
           di.append(index)
           continue
    # K x 4096 matrix
    #mx=np.max(embeddings,axis=0)
    #mn = np.min(embeddings, axis=0)
        me = np.mean(qembedding, axis=0)
    #features = np.concatenate([me, mn, mx])
    #max pool to 1 x 4096
        xtrain = np.append(xtrain, me.reshape(1, -1), axis=0)
    return xtrain,di

df_body_list=df_L['question']
#print(df_body_list[0:10])
X_train=np.empty((0,4096*1))
X_train,di=embedtext(X_train,df_body_list)
print("shape of X train", X_train.shape)
Y = df_L['topic_ID']
Y = to_categorical(Y)
print('Shape of label tensor:', Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
embedvector=4096
classes=13
def model_questionclf(maxlen, classes):
    inp = Input(shape=(maxlen,))
    # if  conv1D is needed

    x=Dense(256,activation='relu')(inp)
    # W  is 256 x 4096, input is 4096 x 1 ==> O=256x1
    outp=Dense(classes,activation='softmax')(x)
    #W  is classes x 256, input is 256 x 1 ==> O=classesx1
    clf_model = Model(inputs=inp, outputs=outp)
    clf_model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    clf_model.summary()
    return clf_model

clf_model=model_questionclf(embedvector,classes)
epochs = 30
batch_size = 32

history = clf_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss', mode='min',patience=30, min_delta=1)])
accr = clf_model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))



ftrain=os.path.join("c:/USHUR/PythonProg/DATAREP/PRIMERICA","test.csv")
df_L=pd.read_csv(ftrain)
df_body_list=df_L['question']
y_test=df_L['topic']
#print(len(df_body_list))
#print(df_body_list)

x_test=np.empty((0,4096*1))
x_test,d=embedtext(x_test,df_body_list)
print(x_test.shape)
y_pred=clf_model.predict(x_test, batch_size=None, verbose=0, steps=None)
#print(type(y_pred),y_pred.shape)
y_index=np.argmax(y_pred,axis=1)
#print("predict array", y_index.shape,y_pred.shape[0])
y_score=np.amax(y_pred,axis=1)
y_mean=np.mean(y_pred,axis=1)
y_var=np.var(y_pred,axis=1)
#print("score array", y_score.shape[0])
accuracy=0
default=0
fp=0
for i in range(y_index.shape[0]):
    #print(y_index[i], y_score[i])
    print(df_body_list[i],topic_d[y_index[i]],y_test[i],y_score[i])


    if y_score[i]<0.3:
        query = df_body_list[i]
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        print("BM25 score", np.amax(doc_scores))
        if np.amax(doc_scores)>2.0:
            if y_test[i]=='low':
                default=default+1
            else:
                fp=fp+1
            continue

    if (topic_d[y_index[i]]==y_test[i]):
        accuracy=accuracy+1
print(accuracy,default,fp,(accuracy+default)/y_index.shape[0])
exit()

def defaultcategory_score(query,threshold=2.0):
    if np.amax(bm25.get_scores(query.split(" ")))>threshold:
        return 1
    else:
        return 0


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

