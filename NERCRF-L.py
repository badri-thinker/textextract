import numpy as np
import os
import string
from nltk.tag import StanfordNERTagger
import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import tree2conlltags

from nltk import ne_chunk
import spacy
import sklearn_crfsuite
import torch
from InferSent import models

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
jar = os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-ner-tagger','stanford-ner.jar')
model3 = os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-ner-tagger/classifiers','english.all.3class.distsim.crf.ser.gz')
model7 = os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-ner-tagger/classifiers','english.muc.7class.distsim.crf.ser.gz')
#jat and models needed for Stanford POS tagger
posjar=os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-pos-tagger','stanford-postagger.jar')
posmodel=os.path.join('c:/USHUR/PythonProg/DATAREP/stanford-pos-tagger/models','english-bidirectional-distsim.tagger')
java_path = "C:/Program Files/Java/jdk1.8.0_191/bin/java.exe"
os.environ['JAVAHOME'] = java_path
nerst = StanfordNERTagger(model7,jar)
#infersent model to encode sentence feature
model_version=2
# modified models.py to overcome stride issue
MODEL_PATH=os.path.join("c:/USHUR/PythonProg/DATAREP/FastText","infersent2.pkl")

params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model= models.InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
W2V_PATH=os.path.join("c:/USHUR/PythonProg/DATAREP/FastText",'crawl-300d-2M.vec')
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)

# encode features for  CRF to  include sentence embedding as a feature

def doc2features(doc, i,s):
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
         'se':s,

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
     return [doc2features(doc[0], i, doc[1]) for i in range(len(doc[0]))]




# labelled data to train CRF fr custom NER tags
print("Start CRF")
data = [(['6432129','is', 'john', 'member', 'no.'], ['CD','IR','PER','B-CID','I-CID']),
(['michael','strahan','member','number', 'is', '4824461'], ['B-PER','I-PER','B-CID','I-CID','IR','CD']),
        (['unable','to','find','member','no','234586'], ['IR','IR' ,'IR', 'B-CID', 'I-CID','CD']),
(['member','joseph','taylor','does','not','have','no'], ['CID','B-PER','I-PER','IR', 'IR', 'IR','IR']),
(['member','no','287586','is','not','working'], ['B-CID','I-CID','CD' ,'IR', 'IR', 'IR']),
(['member','no','186782','belongs','to','barack','obama'], ['B-CID','I-CID','CD' ,'IR', 'IR', 'B-PER','I-PER']),
        (['I', 'have' ,'my' ,'Plan', 'No',  'and', 'Member','name','Larry', 'Blackburn '],['IR','IR',
                                    'IR','B-CID','I-CID','and', 'B-CID' ,'I-CID' ,'B-PER','I-PER'])]




corpus = []
for (doc, tags) in data:
    postags=nltk.pos_tag(doc)
    doc_tag = []
    i=0
    for word, tag in zip(doc,tags):
        doc_tag.append((word,postags[i][1],tag))
        i=i+1
    s=model.encode([" ".join(doc)])
    z=np.where(s>np.mean(s)+np.std(s),'p',(np.where(s<np.mean(s)-np.std(s),'n','0'))).astype('|S1').tostring().decode('utf-8')
#  encode p, n, 0 based on the strength of each dimension

#  z = np.where(model.encode([" ".join(doc)]) >= 0, 'p', 'n').astype('|S1').tostring().decode('utf-8')
#  old encoding, where p is for positive and n for negative
    corpus.append((doc_tag,z))
# create a corpus  that includes a  sentence embedding converted to a string
# CRF does not take vectors of real numbers for feature set
# converted 4096 embedding to a string of p and n (p for positive, n for negative)
# convert  real numbers to  discrete representation string of binary chars (p,n)
# there are other methods to convert embedding to discrete features; we can try them later

X = [extract_features(doc) for doc in corpus]
print("features of X\n",X)


def get_labels(doc):
    return [tag for (token,pos,tag) in doc]


y = [get_labels(doc[0]) for doc in corpus]
#print(y)


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)
fcrf=crf.fit(X,y)
#eli5.show_weights(crf, top=30)
ts1="there are member everywhere"
ts=" david member number is  45678"
ts="the mountain is high. My policy number is 4353535 and India's population is 4443334 davids  member number is 9860300"
s=model.encode(ts)
z=np.where(s>np.mean(s)+np.std(s),'p',(np.where(s<np.mean(s)-np.std(s),'n','0'))).astype('|S1').tostring().decode('utf-8')
test = (nltk.pos_tag(word_tokenize(ts)),z)
# np.where(model.encode([" ".join(ts)]) >= 0, 'p', 'n').astype('|S1').tostring().decode('utf-8') )
# print("POS tags output by nltk")
# print(test[0])
#test = [('member',) ,('number', ),('is',), ('9860300',)]
X_test = extract_features(test)
ans=fcrf.predict_single(X_test)
# print(ts)
# print("NER tags recognized by CRF")
# print(ans)
z=[(x,y)for x,y in zip(word_tokenize(ts),ans)]
# print(z)
desired=list(filter(lambda x:x[1]!='IR',z))

def cw(wttuple):
    return list(zip(wttuple,wttuple[1:],wttuple[2:]))


# context window size is 3
triplet=cw(desired)
# print(triplet)

#relation 1  CID, CD
r1=list(filter(lambda x:(x[0][1]=='B-CID' and x[2][1]=='CD'),triplet))
# print("desired relation")
# print(r1)
r2=[ y[0] for x in r1 for y in x]
# print(r2)
rel2=[]
for x in r1:
    rel=[]
    for y in x:
        rel.append(y[0])
    rel2.append(rel)
print(rel2)
# compare ner tags output by  nltk
ts=" broker Andrew Thompson  wants coverage eff 2/15"
ts1="broker Andrew Thompson  in Portland  wants coverage eff 2/15 Life - 25k & 50K Dental – 100/80/50 1500 Ded," \
    " E/P in basic vision – 10/25 12/12/12 " \
    "They currently have vision & life with UHC (current/renewal Attached)."
ts1=ts1.translate(str.maketrans(dict.fromkeys(' -& ', ' ')))

tokenized_text = word_tokenize(ts1)

pos_nltk = nltk.pos_tag(tokenized_text)
# print("nltk tags")
# print(pos_nltk)
triplet=cw(pos_nltk)
# relation 2  consequent = anything, antecedent = Cardinal Number
r1=list(filter(lambda x:x[2][1]=='CD',triplet))
print("desired relation with CD-NLTK")
# print(r1)
r2=[ y[0] for x in r1 for y in x]
# print(r2)
rel2=[]
for x in r1:
    rel=[]
    for y in x:
        rel.append(y[0])
    rel2.append(rel)
print(rel2)

# print(tree2conlltags(ne_chunk(pos_nltk)))
#nlp = spacy.load('en_core_web_sm')
#doc=nlp(ts1)
#print("spacy ner tags")
#print([(X.text, X.label_) for X in doc.ents])

ner_st= nerst.tag(tokenized_text)
print("stanford ner tags")
print(ner_st)
exit()

