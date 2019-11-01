
import pandas as pd
import os
import nltk


is_noun = lambda pos: pos[:2] == 'NN'
def extractpos(text):
    if type(text)!=str:
        return []
    texts_out=[token[0] for token in nltk.pos_tag(nltk.word_tokenize(text)) if is_noun(token[1])]
    #return token as a list or str(texts_out) if string is desired
    return texts_out
print(extractpos('how do i check on the status of my  renewals'))

def checkinv(query,vocab):
    if len(set(extractpos(query)))==len(set(extractpos(query)).intersection(vocab)):
        return True
    else:
        return False
#build vocab set from questions in the training set.
ftrain=os.path.join("c:/USHUR/PythonProg/DATAREP/PRIMERICA","PrimericaTRAIN.csv")
df_L=pd.read_csv(ftrain,encoding='ISO-8859-1')
df_L['Nouns']=df_L['question'].apply(lambda x:extractpos(x))
df_vocab_list=df_L['Nouns'].tolist()
#print(df_vocab_list)
#convert vocablist to a set
vocab_set= set([item for sublist in df_vocab_list for item in sublist])
if 'tag' in vocab_set:
    print("tag found")
else:
    print ("tag not found")

if 'car' in vocab_set:
    print("car is in vocab")
else:
    print("car is not in vocab")
#  this set can be saved  and read in for deployment
#check if a question contains a OOV words
#example usage
if checkinv('how do i check on the status of my  renewals',vocab_set):
    print('INV')
else:
    print('OOV')
exit()
