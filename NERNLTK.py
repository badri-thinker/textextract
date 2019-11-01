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
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer
from spacy.matcher import PhraseMatcher
import sklearn_crfsuite


jar = os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-ner-tagger','stanford-ner.jar')
# 3 and 7 NER classes model needed for stanford NER tagger
model = os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-ner-tagger/classifiers','english.all.3class.distsim.crf.ser.gz')
model7 = os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-ner-tagger/classifiers','english.muc.7class.distsim.crf.ser.gz')
#jat and models needed for Stanford POS tagger
posjar=os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-pos-tagger','stanford-postagger.jar')
posmodel=os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-pos-tagger/models','english-bidirectional-distsim.tagger')
java_path = "C:/Program Files/Java/jdk1.8.0_191/bin/java.exe"
os.environ['JAVAHOME'] = java_path


def doc2features(doc, i):
     word = doc[i][0]
     postag=doc[i][1]
     # Features from current word
     features = {
         'word.word': word,
         'word.lower()': word.lower(),
         'word[-3:]': word[-3:],
         'word[-2:]': word[-2:],
         'word.isupper()': word.isupper(),
         'word.istitle()': word.istitle(),
         'word.isdigit()': word.isdigit(),
         'postag': postag,
         'postag[:2]': postag[:2],
     }
     # Features from previous word
     if i > 0:
         word1 = doc[i - 1][0]
         postag1 = doc[i - 1][1]
         features.update({
             '-1:word.lower()': word1.lower(),
             '-1:word.istitle()': word1.istitle(),
             '-1:word.isupper()': word1.isupper(),
             '-1:postag': postag1,
             '-1:postag[:2]': postag1[:2],
         })

     else:
          features['BOS'] = True  # Special "Beginning of Sequence" tag

     # Features from next word
     if i < len(doc) - 1:
         word1 = doc[i + 1][0]
         postag1 = doc[i + 1][1]
         features.update({
             '+1:word.lower()': word1.lower(),
             '+1:word.istitle()': word1.istitle(),
             '+1:word.isupper()': word1.isupper(),
             '+1:postag': postag1,
             '+1:postag[:2]': postag1[:2],
         })

     else:
          features['EOS'] = True  # Special "End of Sequence" tag
     return features


def extract_features(doc):
     return [doc2features(doc, i) for i in range(len(doc))]



st = StanfordNERTagger(model7,jar)
post=StanfordPOSTagger(posmodel,posjar)

text1 = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'
text2="Please advise on the options the deceased clients wife has in relation to this pension" \
     "   She wishes to exercise ARF option if available "
text="Hi I was trying to register online but I was n t recognised " \
     "  My  membership number is 4824461  and the eff is 1.1.2019    " \
     "Looking to register on Pension Planet Robert Manning" \
     "   but Irish Ronnie Gardner member website ca n t find my details        " \
     "Richard Wade.   Not recognised when trying to register. My member number " \
     "is 9860300   Christina Wood Kim West     and I just tried to register for pension planet  " \
     " but always got the result      not recognised     " \
     "Are you able to help me so I can register      Thanks a mill already"

tokenized_text = word_tokenize(text)
ner_st= st.tag(tokenized_text)
#print(ner_st)

pos_st=post.tag(tokenized_text)
#print(pos_st)

pos_nltk = nltk.pos_tag(tokenized_text)
print(pos_nltk)
custid=[(x ,y[0]) for x, y in enumerate(pos_nltk) if y[1] == 'CD']
print(custid)
for x,y in enumerate(custid):
     print(pos_nltk[y[0]-2][0],pos_nltk[y[0]-1][0],y[1])
#tetxblob provides  a way to extract pos tags
print("Start CRF")
data = [(['6432129','is,', 'my', 'member', 'no.',], ['IR','IR','IR','CID','IR']),
(['my','member','number', 'is', '4824461'], ['IR','CID','IR','IR','CD']),
        (['unable','to','find','member','no','234586'], ['IR','IR' ,'IR', 'CID', 'IR','CD']),
(['member','no','287586','is','not','working'], ['CID','IR','CD' ,'IR', 'IR', 'IR'])
        ]


corpus = []
for (doc, tags) in data:
    postags=nltk.pos_tag(doc)
    doc_tag = []
    i=0
    for word, tag in zip(doc,tags):
        doc_tag.append((word,postags[i][1],tag))
        i=i+1
    corpus.append(doc_tag)
print(corpus)

X = [extract_features(doc) for doc in corpus]
print("features of X\n",X)


def get_labels(doc):
    return [tag for (token,pos,tag) in doc]


y = [get_labels(doc) for doc in corpus]
print(y)


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)
fcrf=crf.fit(X,y)
ts="Are you a  member"
#ts="member number is 9860300"
test = nltk.pos_tag(word_tokenize(ts))
print(test)
#test = [('member',) ,('number', ),('is',), ('9860300',)]
X_test = extract_features(test)
ans=fcrf.predict_single(X_test)
print(ans)

exit()










blob= TextBlob(text)
print(blob.tags)
custid=[(x ,y[0]) for x, y in enumerate(blob.tags) if y[1] == 'CD']
#extract  index and value of  tags whose  second attr is CD
print(custid)
#enumerate  prefix of tuples and its txt  of CD tags
for x,y in enumerate(custid):
     print(blob.tags[y[0]-2][0],blob.tags[y[0]-1][0],y[1])

#same as above using spaCy
nlp = spacy.load('en_core_web_sm')
text2="My name is albert and we are in pittsburgh tonight"
doc=nlp(text)
print([(X.text, X.label_) for X in doc.ents])
label="customerid"
matcher=spacy.matcher.PhraseMatcher(nlp.vocab)
for i in ['membership', 'member','policy']:
     matcher.add(label,None,nlp(i))
     
one=nlp(text)
matches=matcher(one)
y=[x for x in matches]
print(y)
print([(X.text, X.label_) for X in one.ents])
exit()
postags=[]
for token in doc:
    postags.append(tuple((token.text, token.pos_)))
print(postags)

custid=[(x ,y[0]) for x, y in enumerate(postags) if y[1] == 'NUM']
print(custid)
for x,y in enumerate(custid):
     print(postags[y[0]-2][0],postags[y[0]-1][0],y[1])
exit()


doc = doc(nlp.vocab, [u'rats', u'make', u'good', u'pets'])
gold = GoldParse(doc, [u'U-ANIMAL', u'O', u'O', u'O'])
ner = EntityRecognizer(nlp.vocab, entity_types=['ANIMAL'])
ner.update(doc, gold)

nlp = spacy.load('en')

#doc = nlp('WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement.')
doc=nlp(text)
print([ent for ent in doc.ents])
print([ent for ent in doc.noun_chunks])

print([(X.text, X.label_) for X in doc.ents])
spans = list(doc.ents) + list(doc.noun_chunks)
for span in spans:
        span.merge()
print(span)
exit()