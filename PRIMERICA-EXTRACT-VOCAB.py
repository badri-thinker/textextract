import numpy as np
import pandas as pd
import os
import spacy
import nltk
is_noun = lambda pos: pos[:2] == 'NN'
def extractpos(text):
    if type(text)!=str:
        return []
    texts_out=[token[0] for token in nltk.pos_tag(nltk.word_tokenize(text)) if is_noun(token[1])]
    #return token as a list or str(texts_out) if string is desired
    return texts_out
'''
#nlp = spacy.load('en_core_web_sm')
def extractpos(text):
    allowed_postags=['NOUN']
    # if other POS TAGS are needed 'ADJ','ADP','ADV','AUX', 'DET','NOUN', 'PART', 'VERB']
    doc = nlp(text)
    texts_out=[token.text for token in doc if token.pos_ in allowed_postags]
    #return token as a list or str(texts_out) if string is desired
    return texts_out
'''
def checkoov(query,vocab):
    return set(extractpos(query)).intersection(vocab)
#build vocab set from questions in the training set.
ftrain=os.path.join("c:/USHUR/PythonProg/DATAREP/PRIMERICA","QTRAIN.csv")
df_L=pd.read_csv(ftrain)
df_L['Nouns']=df_L['question'].apply(lambda x:extractpos(x))
df_vocab_list=df_L['Nouns'].tolist()
print(len(df_vocab_list))
#convert vocablist to a set
vocab_set= set([item for sublist in df_vocab_list for item in sublist])
#  this set can be saved  and read in for deployment
#check if a question contains a OOV words
if checkoov('How do i bake a cake',vocab_set):
    print('INV')
else:
    print('OOV')
exit()
