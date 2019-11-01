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

jar = os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-ner-tagger','stanford-ner.jar')
# 3 and 7 NER classes model needed for stanford NER tagger
model = os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-ner-tagger/classifiers','english.all.3class.distsim.crf.ser.gz')
model7 = os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-ner-tagger/classifiers','english.muc.7class.distsim.crf.ser.gz')
#jat and models needed for Stanford POS tagger
posjar=os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-pos-tagger','stanford-postagger.jar')
posmodel=os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-pos-tagger/models','english-bidirectional-distsim.tagger')
java_path = "C:/Program Files/Java/jdk1.8.0_191/bin/java.exe"
os.environ['JAVAHOME'] = java_path

st = StanfordNERTagger(model7,jar)
post=StanfordPOSTagger(posmodel,posjar)

text1 = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'
text2="Please advise on the options the deceased clients wife has in relation to this pension" \
     "   She wishes to exercise ARF option if available "
text="Hi I was trying to register online but I was n t recognised " \
     "  My  France number is 4824461      " \
     "Looking to register on Pension Planet Robert Manning" \
     "   but Irish Ronnie Gardner website ca n t find my details        " \
     "Richard Wade "
text='How can I pay my car renewal'
tokenized_text = word_tokenize(text)
ner_st= st.tag(tokenized_text)
print(ner_st)

pos_st=post.tag(tokenized_text)
print(pos_st)
exit()
pos_nltk = nltk.pos_tag(tokenized_text)
print(pos_nltk)

blob= TextBlob(text)
print(blob.tags)
print("tree stanford\n")
print("type of chunk",type(ne_chunk(pos_st)))

print("type of tree",len(tree2conlltags(ne_chunk(pos_st))))
print("tree nltk\n")
print(tree2conlltags(ne_chunk(pos_nltk)))
print("tree blob\n")
print(ne_chunk(pos_nltk))
print(tree2conlltags(ne_chunk(blob.tags)))
exit()