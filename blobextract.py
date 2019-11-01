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
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from scipy.stats import zscore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
analyzer = SentimentIntensityAnalyzer()
#print(analyzer.polarity_scores("this is an awesome movie. I hated it")['compound'])

testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")
print(testimonial.sentiment.polarity*(1-testimonial.sentiment.subjectivity))
exit()