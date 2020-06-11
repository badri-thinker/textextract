import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import ast
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Embedding, Flatten, Softmax,SpatialDropout1D, LSTM, Bidirectional, GRU, Dropout, TimeDistributed, Activation
from keras import Sequential, initializers, regularizers, constraints,optimizers
from keras.backend import squeeze,dot,expand_dims,tanh,sum, cast, epsilon,floatx, exp
from keras.layers import Dense, Input,  Layer, Lambda, Reshape, Conv2D, MaxPool2D, Concatenate
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from InferSent import models as Inf
import torch
import xml.etree.ElementTree as et
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from gensim.models import KeyedVectors
analyzer = SentimentIntensityAnalyzer()
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import spacy
import re
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from spellchecker import SpellChecker
nlp = spacy.load('en_core_web_md')
spell=SpellChecker(distance=1)
lm=WordNetLemmatizer()
from laserembeddings import Laser
laser=Laser()
'''
me=np.random.random((5,4))
print(me.shape)
for input_t in me:
    print(input_t)


'''
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Confusion matrix with Normalization'
        else:
            title = 'Confusion matrix without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix without normalization')
    print(cm)

    fig, ax = plt.subplots(figsize=(2*cm.shape[0],2*cm.shape[0]))
    im = ax.matshow(cm,  cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    #plt.rc('xtick', labelsize=20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=10)
    ax.xaxis.tick_bottom()
    plt.yticks(tick_marks, classes, size=10)
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             #rotation_mode="anchor",size=12)
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment='center', verticalalignment='center',
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax

class AttentionM(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix.
    et=tanh(W.ht + b)
    at=soft(et)
    Ot=SIGMA (at.ht)

    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)
    # Usage
        enc = LSTM(EMBED_SIZE, return_sequences=True)(...)
        att = AttentionM()(enc)
    """

    def __init__(self, **kwargs):
        super(AttentionM, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS,)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionM, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, OUTPUT_FEATURE_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # atx: (BATCH_SIZE, MAX_TIMESTEPS, 1)
        atx = K.expand_dims(at, axis=-1)
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, OUTPUT_FEATURE_SIZE)
        ot = x * atx
        # output: (BATCH_SIZE, OUTPUT_FEATURE_SIZE)
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionM, self).get_config()
class AttentionMV(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix
    and a user provided context vector.
 et=tanh(U.C+ W.ht + b)
    at=soft(et)
    Ot=SIGMA (at.ht)
    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ctx - 2D Tensor of shape (BATCH_SIZE, EMBED_SIZE_CTX) (optional)

    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)
    # Usage
        enc = Bidirectional(GRU(EMBED_SIZE,return_sequences=True))(...)
        # with user supplied vector
        ctx = GlobalAveragePooling1D()(enc)
        att = AttentionMV()([enc, ctx])

    """

    def __init__(self, **kwargs):
        super(AttentionMV, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type(input_shape) is list and len(input_shape) == 2
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS, 1)
        # U: (EMBED_SIZE_CV, MAX_TIMESTEPS)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[0][-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[0][1], 1),
                                 initializer="zeros")
        self.U = self.add_weight(name="U_{:s}".format(self.name),
                                 shape=(input_shape[1][-1],
                                        input_shape[0][1]),
                                 initializer="normal")
        super(AttentionMV, self).build(input_shape)

    def call(self, xs, mask=None):
        # input: [x, u]
        # x: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # c: (BATCH_SIZE, EMBED_SIZE_CV)
        # U: (BATCH_SIZE, EMBED_SIZE_CV,MAX_TIMESTEPS)
        x, c = xs
        # et: (BATCH_SIZE, MAX_TIMESTEPS)`
        # __,embedding_cv * embedding_cv x timesteps= ___, timesteps
        #print("array shapes",self.U.shape,self.W.shape,c.shape,x.shape)
        et = K.dot(c, self.U) + K.squeeze((K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None and mask[0] is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE) basically a weighted pooling of output
        print("k.sum", type(K.sum(ot, axis=1)))
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (input_shape[0][0], input_shape[0][-1])

    def get_config(self):
        return super(AttentionMV, self).get_config()

def model_lstm_infersent_doc(max_classes,embed_size):
    # entire document is avg/max/min pooled into one vector
    inp = Input(shape=(embed_size,))
    x = Dense(64, activation="relu")(inp)
    #,kernel_regularizer=regularizers.l2(0.003),activity_regularizer=regularizers.l1(0.01))(x)
    #x = Dropout(0.5)(x)
    x = Dense(max_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model





def LaserSubjEmbedding(EM,data_list):
    #returns a 2D tensor (samples, sentence embedding)
    #print("datalist", type(data_list))
    #print("datalist",data_list.shape)
    for index, s in enumerate(data_list):
        if len(s) == 0:
            me = np.random.rand(1024)
            #print("list is empty",index,me.shape)
            EM = np.append(EM, me.reshape(1,-1), axis=0)
        #print(index)
            continue

        #s=sent_tokenize(s)
        #print(type(s), len(s), s)
        try:
            doc = nlp(s)
            s = [sent.string.strip() for sent in doc.sents]
            embedding = laser.embed_sentences(s,lang='en')
            print(index)
            me = np.mean(embedding, axis=0)
            #print("mean",me.shape)
        except:
            me = np.random.rand(1024)
            print("exception",index,me.shape)
            EM = np.append(EM, me.reshape(1,-1), axis=0)
            continue
        EM =np.append(EM, me.reshape(1, -1), axis=0)
        #print(EM.shape)
    return EM


fotrain=os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK","unumuk.csv")
fosubj=os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK","unumuksubjall.csv")
foresults=os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK","results.txt")
fr=open(foresults,'a+')
#fntrain=os.path.join("c:/USHUR/PythonProg/DATAREP/PRIMERICA","QTRAIN-NEW.csv")
df_L=pd.read_csv(fotrain)
print(df_L.head())
print(df_L.info())
df_L=df_L.dropna()
print(df_L.head())
print(df_L.info())
df_L['received']=df_L['received'].apply(lambda x: ast.literal_eval(x))
df_L['sent']=df_L['sent'].apply(lambda x: ast.literal_eval(x))
#,converters={'received':ast.literal_eval,'sent':ast.literal_eval})

#code for contex embedding of Subect line
print(type(df_L.loc[0,'phrase']))
def extsubj(los,exts):
    #print(type(los))
    subjlines=exts.findall(los)
    return ' '.join(subjlines)

exts=re.compile(r'Subject :(.*?)\s\s')
df_L['subject']=df_L.loc[:,'phrase'].apply(lambda x:extsubj(x,exts))
headers=["topic","subject"]
df_L.to_csv(fosubj,columns=headers)
embed_size=1024
XC=np.zeros((0,embed_size))
#XC=LaserSubjEmbedding(XC,df_L['phrase'].values)
#print(XC.shape)

foXC=os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK","XC.dat")
#XC.dump(foXC)
XC=np.load(foXC,allow_pickle=True)

Y = pd.get_dummies(df_L['topic'].values)
print('Shape of label tensor:', Y.shape)
print(type(Y))
df_L['topic']=df_L['topic'].astype('category')
df_L['topic_v']=df_L['topic'].cat.codes
topic_d = dict(enumerate(df_L['topic'].cat.categories))
inverted_topic_d=dict(map(reversed, topic_d.items()))
list_of_topics=np.asarray(df_L['topic'].unique().tolist())
Y=df_L['topic_v'].values
XC_train, XC_test,Y_train,Y_test= train_test_split(XC,Y,test_size = 0.10, random_state = 42)

lr=LogisticRegression()
flr=lr.fit(XC_train, Y_train)
y_pred=flr.predict(XC_test)
#print(type(y_pred),y_pred,y_test)
print("LogReg accuracy",np.mean(y_pred==Y_test))

svr= LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)

#svr=SVC(kernel='rbf', random_state=0, gamma=.1, C=10)
fsvr=svr.fit(XC_train, Y_train)

y_pred=fsvr.predict(XC_test)
print("SVM accuracy",np.mean(y_pred==Y_test))
cm = confusion_matrix(Y_test, y_pred)
print(cm)
exit()



epochs = 50
batch_size = 64
actual_no_threads=2
#def model_lstm_SR_Words(max_no_threads, max_features_r, max_features_s,max_classes, maxlen, embed_size, embedding_matrix_r,embedding_matrix_s,bs
model=model_lstm_infersent_doc(len(topic_d), embed_size)
y_ints = np.argmax(Y_train.values,axis=1)
#convert labels into array
cl=np.array(list(inverted_topic_d.values()))

class_weights = class_weight.compute_class_weight('balanced',
                                                 cl,
                                                 y_ints)
history = model.fit(XC_train,Y_train,class_weight=class_weights,epochs=epochs,batch_size=batch_size,validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', mode='min',patience=8, min_delta=0.01)])

accr = model.evaluate(XC_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
#print('Score {:.2f}\n').format(score)
y_pred=model.predict(XC_test,batch_size=None, verbose=0, steps=None)
print(y_pred.shape,y_pred)
Y_pred=np.argmax(y_pred,axis=1)
print(Y_pred.shape)
print(np.argmax(Y_test.values,axis=1).shape)

accuracy = accuracy_score(np.argmax(Y_test.values,axis=1), Y_pred)
print('Accuracy: %f' % accuracy)
fr.write('Accuracy: %f \n' % accuracy)

cm = confusion_matrix(np.argmax(Y_test.values,axis=1), Y_pred)
print(cm)
print(topic_d.values())
plot_confusion_matrix(np.argmax(Y_test.values,axis=1),Y_pred,classes=topic_d.values(),normalize=True,
                     title='Confusion matrix, with normalization')
plt.show()
print(history.history.keys())
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.show()
exit()



'''
XC=np.zeros((0,4096))
XC,di=InferSubjEmbedding(XC,df_L['subject'].values)
print(XC.shape,di)
'''
exit()



'''
for i in  range(10):
    phrase=df_L.loc[i,'subject']
    print(type(phrase))
    print(phrase)
    #print(exts.findall(sent_tokenize(phrase)[0]))
    #try:
       # print(sent_tokenize(phrase)[1])
    #except: KeyError
'''


#df = pd.read_csv("in.csv",converters={"Col3": literal_eval}to convert string to list)

#path for saving google embedding of training data
foembed_r_and_s=os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK","google_r_and_s.dat")
foembed_s=os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK","google_s.dat")



fig1,axes1=plt.subplots(1,1,figsize=(8,6))
plt.xticks(fontsize=6)
plt.rc('ytick', labelsize=20)

df_L['topic']=df_L['topic'].astype('category')
topic_d = dict(enumerate(df_L['topic'].cat.categories))
inverted_topic_d=dict(map(reversed, topic_d.items()))
list_of_topics=np.asarray(df_L['topic'].unique().tolist())
df_L.topic.value_counts().reindex(list_of_topics).plot(kind="bar",title="Original Frequency of topics",ax=axes1)
#plt.show()

print(topic_d)
print(inverted_topic_d)


'''
r=df_L['received'].values
s=df_L['sent'].values
print(type(r[0]))


#print(ast.literal_eval(x[2]))
for  i in range(r.shape[0]):
    print("number of receiver threads",len(r[i]))
    for n in r[i]:
        print(n.strip())
    print("number of send threads", len(s[i]))
    for n in s[i]:
        print(n.strip())
    print("next email")
'''
MAX_NB_WORDS = 50000
# Max number of words in each application.
MAX_SEQUENCE_LENGTH = 200
#M max number of threads
MAX_NO_THREADS=5
# This is 200 for GLOVE and 300 for Google
EMBEDDING_DIM = 300
# number fo classes
NUM_CLASSES=len(topic_d)
r=df_L['received'].values
s=df_L['sent'].values
print("r series",type(r[0]),type(r[0][0]),r.shape)

X_R=np.zeros((r.shape[0],MAX_NO_THREADS),dtype=object)
X_S=np.zeros((s.shape[0],MAX_NO_THREADS),dtype=object)
X_R_N=np.zeros((r.shape[0],MAX_NO_THREADS,MAX_SEQUENCE_LENGTH),dtype=int)
X_S_N=np.zeros((s.shape[0],MAX_NO_THREADS,MAX_SEQUENCE_LENGTH),dtype=int)
#X_RT=np.zeros((r.shape[0]),dtype=object)
#X_ST=np.zeros((s.shape[0],MAX_SEQUENCE_LENGTH))

count=0
print("length of list in sample",len(r[0]),type(r[0]))

for i in range(r.shape[0]):
    r[i] = (r[i] + MAX_NO_THREADS *[' '])[:MAX_NO_THREADS]
    s[i] = (s[i] + MAX_NO_THREADS * [' '])[:MAX_NO_THREADS]
#append the string with  MAX_NO_THREADS blank items and then cut it off at MAX_NO_THREADS
print(type(r),len(r[0]))
print(r[0][0])
print(type(r[0][0]))


for i in range(r.shape[0]):
    for j in range(MAX_NO_THREADS):
        X_R[i,j]=r[i][j]
        X_S[i,j]=s[i][j]
print(X_R.shape)
print(type(X_R[0,0]))


tokenizer= Tokenizer(num_words=MAX_NB_WORDS)
#, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
#for thread in range(MAX_NO_THREADS):
tokenizer.fit_on_texts(np.vstack((X_R,X_S)).ravel())
word_index = tokenizer.word_index
print('Found %s unique token in both R and S ' % len(word_index))

for j in range(MAX_NO_THREADS):
    X_R[:, j] = tokenizer.texts_to_sequences(X_R[:, j])
    X_R_N[:,j]= pad_sequences(X_R[:,j], padding='post',maxlen=MAX_SEQUENCE_LENGTH)


for j in range(MAX_NO_THREADS):
    X_S[:,j]=tokenizer.texts_to_sequences(X_S[:,j])
    X_S_N[:,j]= pad_sequences(X_S[:,j], padding='post',maxlen=MAX_SEQUENCE_LENGTH)

for i in range(10):
    print("receive", df_L.loc[i, 'topic'], r[i][0])
    print("send",df_L.loc[i,'topic'],s[i][0])
    #print("send ",s[i][1])
    print("send",X_S[i,0])
    print("send", X_S[i, 1])
    print("send", X_R[i, 0])
    print("send", X_R[i, 1])
    #print("send",X_S_N[i,0])


print('Shape of received data tensor:', X_R_N.shape)
print('Shape of received data tensor:', X_R_N[0].shape)
print('Shape of sent  data tensor:', X_S_N.shape)
print('Shape of sent  data tensor:', X_S_N[0].shape)

Y = pd.get_dummies(df_L['topic'].values)
print('Shape of label tensor:', Y.shape)
print(type(Y))

X_R_train, X_R_test,X_S_train,X_S_test ,XC_train,XC_test,Y_train,Y_test= train_test_split(X_R_N,X_S_N,XC,Y,test_size = 0.10, random_state = 42)
#Y_train, Y_test=train_test_split(Y,test_size = 0.20, random_state = 42)
print(X_R_train.shape,X_S_train.shape,XC_train.shape,Y_train.shape)
print(X_R_test.shape,X_S_test.shape,XC_test.shape,Y_test.shape)
print(Y_train.shape,Y_test.shape)


#for google, embedding dim is 300
filename = os.path.join("c:/USHUR/PythonProg/DATAREP",'GoogleNews-vectors-negative300.bin')
w2v_model = KeyedVectors.load_word2vec_format(filename, binary=True)


e_f=0
def get_embedding_matrix(embedding_matrix,word_index,e_f):
    for word, i in word_index.items():
        try:
            w2v_model[[word.lower()]]
            embedding_matrix[i]=w2v_model[[word.lower()]]
            e_f=e_f+1
        except KeyError:
            try:
                w2v_model[[spell.correction(lm.lemmatize(word.lower()))]]
                embedding_matrix[i] = w2v_model[[spell.correction(lm.lemmatize(word.lower()))]]
                e_f = e_f+ 1
            except KeyError:
                continue
    return embedding_matrix,e_f


#if embedding matrix is not already computed
#embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
#embedding_matrix,e_f=get_embedding_matrix(embedding_matrix,word_index,e_f)
#print("number of embeddings found",e_f)
#embedding_matrix = np.zeros((len(word_index_s) + 1, EMBEDDING_DIM))
#embedding_matrix_s=get_embedding_matrix(embedding_matrix,word_index_s)



# save matrix for future use if needed
#embedding_matrix.dump(foembed_r_and_s)
#embedding_matrix_s.dump(foembed_s)
# load previously saved matrix if available
embedding_matrix=np.load(foembed_r_and_s,allow_pickle=True)
#embedding_matrix_s=np.load(foembed_S,allow_pickle=True)




epochs = 50
batch_size = 64
actual_no_threads=3
#def model_lstm_SR_Words(max_no_threads, max_features_r, max_features_s,max_classes, maxlen, embed_size, embedding_matrix_r,embedding_matrix_s,bs
model=model_lstm_INFERSENT(MAX_NO_THREADS,actual_no_threads,len(word_index)+1,len(topic_d),MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,
                          embedding_matrix,batch_size)

#classweight={0:1.7,1:1.0}
history = model.fit([X_R_train, X_S_train,XC_train],Y_train,epochs=epochs, batch_size=batch_size,validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', mode='min',patience=5, min_delta=0.01)])

accr = model.evaluate([X_R_test,X_S_test,XC_test],Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
#print('Score {:.2f}\n').format(score)
y_pred=model.predict([X_R_test, X_S_test,XC_test],batch_size=None, verbose=0, steps=None)
print(y_pred.shape,y_pred)
Y_pred=np.argmax(y_pred,axis=1)
print(Y_pred.shape)
print(np.argmax(Y_test.values,axis=1).shape)

accuracy = accuracy_score(np.argmax(Y_test.values,axis=1), Y_pred)
print('Accuracy: %f' % accuracy)
fr.write('Accuracy: %f \n' % accuracy)

cm = confusion_matrix(np.argmax(Y_test.values,axis=1), Y_pred)
print(cm)

plot_confusion_matrix(np.argmax(Y_test.values,axis=1),Y_pred,classes=topic_d.values(),normalize=True,
                     title='Confusion matrix, with normalization')
plt.show()
print(history.history.keys())
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.show()
exit()


h = K.variable(value=np.random.normal(size=(1,1)))
c = K.variable(value=np.random.normal(size=(1,1)))
x1=K.placeholder(shape=(1,1))
send_t = Input(shape=(3,4,1,))
recv_t= Input(shape=(3,4,1,))
print("type of send",send_t[:,0],send_t[:,1].shape)
#a sample consists of  two alternating paragraphs (S1, R1, S2, R2 etc)
for input_t in range(3):
    #splitting of tensor needs to be done at the keras layer using LAMBDA
    inps = Lambda(lambda x:x[:,input_t])(send_t)
    inpr = Lambda(lambda x:x[:,input_t])(recv_t)
    x ,h1,c1=LSTM(1,return_sequences=False,return_state=True)(inps,initial_state=[h,c])
    x1 ,h,c=LSTM(1,return_sequences=False,return_state=True)(inpr,initial_state=[h1,c1])

model = Model(inputs=[send_t,recv_t], outputs=[x1,h,c])
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
data1 = np.array([[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5],[0.3,0.4,0.5,0.6]]).reshape((1,3,4,1))
data2 = np.array([[0.4,0.5,0.6,0.7],[0.6,0.7,0.8,0.9],[0.5,0.6,0.7,0.8]]).reshape((1,3,4,1))
# make and show prediction
print(model.predict([data1,data2]))
exit()
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

fo=os.path.join("d:/USHUR/PythonProg/DATAREP/sentiment","Datafiniti_hotel_Reviews_L.csv")
foout=os.path.join("d:/USHUR/PythonProg/DATAREP/sentiment","Datafiniti_title_text_rating.csv")
df_L1=pd.read_csv(fo)
df_L1=df_L1.rename(columns={'reviews.rating':'rating','reviews.text':'text','reviews.title':'title'})
df_L1=df_L1[['rating','text','title']]
df_L1.dropna(inplace=True)
df_L1.info()
print(df_L1.head())
df_L1.to_csv(foout)
#df_L1['text']=df_L1.apply(lambda x: x['title'] + ' ' + x['text'],axis=1)
df_C=df_L1['title']
#print(df_L1['text'].head())
def ns(rating):
    if rating in [1,2]:
        return 'negative'
    else:
        if rating in [3,4]:
            return 'neutral'
        else:
            return 'positive'

df_L1['rating']=df_L1['rating'].apply(lambda x:ns(x))
'''
fotrain1=os.path.join("d:/USHUR/PythonProg/DATAREP/sentiment","Datafiniti_Hotel__L_2C.csv")
#fo_reviews_train=os.path.join("d:/USHUR/PythonProg/DATAREP/sentiment","reviews.tsv")
#df_L=pd.read_csv(fo_reviews_train,delimiter='\t',encoding='windows-1252')
fotrain2=os.path.join("d:/USHUR/PythonProg/DATAREP/sentiment","Tweets.csv")
df_L1=pd.read_csv(fotrain1)
df_L1=df_L1.rename(columns={'reviews.rating':'rating','reviews.text':'text'})


df_L1['rating']=df_L1['rating'].apply(lambda x:ns(x))
df_L2=pd.read_csv(fotrain2,encoding='ISO-8859-1')
df_L=pd.concat([df_L1,df_L2],ignore_index=True)
df_L.info()
df_L=df_L[['rating','text']]
df_L.dropna(inplace=True)
df_L.info()
df_L.to_csv(fo)
'''
# for tab separated file delimiter='\t',encoding='windows-1252')

#df_L.info()
#df_L.dropna(inplace=True)
#df_L=df_L[['airline_sentiment','text']]
#print(df_L.head())

#df_L=df_L[['rating','review']]
#df_L.info()
#labels=set(df_L['rating'].tolist())
#print(labels)
#print(df_L.head())

def SS (type,col1,col2):
    if type==1:
        return analyzer.polarity_scores(str(col1 + col2))['compound']
    else:
        return blobsenti(str(col1 + col2))
'''
df_L["VScore"] = df_L.apply(lambda x:SS(1,x['title'],x['text']),axis=1)
df_L["TBScore"] = df_L.apply(lambda x: SS(2,x['title'] ,x['text']),axis=1)
df_L["VScoreT"] = df_L.apply(lambda x:SS(1,x['text'],' '),axis=1)
df_L["TBScoreT"] = df_L.apply(lambda x: SS(2,x['text'] ,' '),axis=1)
df_L=df_L[['rating','VScore','VScoreT','TBScore','TBScoreT','title','text']]
df_L.to_csv(fo)
'''

model_version=2
# modified models.py
MODEL_PATH=os.path.join("c:/USHUR/PythonProg/DATAREP/FastText","infersent2.pkl")

params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model= Inf.InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
W2V_PATH=os.path.join("c:/USHUR/PythonProg/DATAREP/FastText",'crawl-300d-2M.vec')
model.set_w2v_path(W2V_PATH)
# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)





def InferDocEmbedding(EM,data_list):
    #returns a 2D tensor (samples, sentence embedding)
    di=[]
    for index, s in enumerate(data_list):
    #print(type(s))
        if type(s) is not str:
            di.append(index)
            me=np.random.rand(4096)
            EM = np.append(EM, me, axis=0)
            continue
        if len(s) == 0:
            me = np.random.rand(4096)
            EM = np.append(EM, me, axis=0)
            di.append(index)
        #print(index)
            continue

        s=sent_tokenize(s)
        #print(type(s), len(s), s)
        try:
            embedding = model.encode(s, tokenize=True)
            me = np.mean(embedding, axis=0)
        except:
            di.append(index)
            me = np.random.rand(4096)
            EM = np.append(EM, me, axis=0)
            continue
        EM =np.append(EM, me.reshape(1, -1), axis=0)
    return EM, di



def InfersentEmbedding(EM, MAX_SENT,data_list):
    #returns a 3D tensor (samples, sentences, sentence embedding
    di=[]
    for index, s in enumerate(data_list):
    #print(type(s))
        if type(s) is not str:
            di.append(index)
            continue
        if len(s) == 0:
            di.append(index)
        #print(index)
            continue
        doc = nlp(s)
        s = [sent.string.strip() for sent in doc.sents]
        #s=sent_tokenize(s)
        if len(s)>MAX_SENT:
            s=s[0:MAX_SENT]
        print(type(s), len(s), s)
        try:
            embedding = model.encode(s, tokenize=True)
            #print(embedding.shape)
        except:
            print("cant find embedding")
            di.append(index)
            continue
        embeddings=np.zeros((MAX_SENT,4096))
        embeddings[0:embedding.shape[0],:]=embedding
        #print(EM.shape,embeddings.shape,np.array([embeddings]).shape)
        EM = np.append(EM,np.array([embeddings]),axis=0)
    return EM, di


class AttentionM(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix.
    et=tanh(W.ht + b)
    at=soft(et)
    Ot=SIGMA (at.ht)

    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)
    # Usage
        enc = LSTM(EMBED_SIZE, return_sequences=True)(...)
        att = AttentionM()(enc)
    """

    def __init__(self, **kwargs):
        super(AttentionM, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS,)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionM, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # atx: (BATCH_SIZE, MAX_TIMESTEPS, 1)
        atx = K.expand_dims(at, axis=-1)
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ot = x * atx
        # output: (BATCH_SIZE, EMBED_SIZE)
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionM, self).get_config()


class AttentionMV(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix
    and a user provided context vector.
 et=tanh(U.C+ W.ht + b)
    at=soft(et)
    Ot=SIGMA (at.ht)
    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ctx - 2D Tensor of shape (BATCH_SIZE, EMBED_SIZE_CTX) (optional)

    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)
    # Usage
        enc = Bidirectional(GRU(EMBED_SIZE,return_sequences=True))(...)
        # with user supplied vector
        ctx = GlobalAveragePooling1D()(enc)
        att = AttentionMV()([enc, ctx])

    """

    def __init__(self, **kwargs):
        super(AttentionMV, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type(input_shape) is list and len(input_shape) == 2
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS, 1)
        # U: (EMBED_SIZE_CV, MAX_TIMESTEPS)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[0][-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[0][1], 1),
                                 initializer="zeros")
        self.U = self.add_weight(name="U_{:s}".format(self.name),
                                 shape=(input_shape[1][-1],
                                        input_shape[0][1]),
                                 initializer="normal")
        super(AttentionMV, self).build(input_shape)

    def call(self, xs, mask=None):
        # input: [x, u]
        # x: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # c: (BATCH_SIZE, EMBED_SIZE_CV)
        # U: (BATCH_SIZE, EMBED_SIZE_CV,MAX_TIMESTEPS)
        x, c = xs
        # et: (BATCH_SIZE, MAX_TIMESTEPS)`
        # __,embedding_cv * embedding_cv x timesteps= ___, timesteps
        #print("array shapes",self.U.shape,self.W.shape,c.shape,x.shape)
        et = K.dot(c, self.U) + K.squeeze((K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None and mask[0] is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE) basically a weighted pooling of output
        print("k.sum", type(K.sum(ot, axis=1)))
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (input_shape[0][0], input_shape[0][-1])

    def get_config(self):
        return super(AttentionMV, self).get_config()


def model_lstm_SR_Sentences(max_no_threads,max_no_sent, max_classes,embed_size, batch_size):
    send_t = Input(shape=(max_no_threads,max_no_sent, embed_size,))
    recv_t = Input(shape=(max_no_threads, max_no_sent, embed_size,))

    #initalize SLTM state
    #import keras.backend as K
    h_s = K.variable(value=np.random.normal(size=(batch_size, 128)))
    c_s = K.variable(value=np.random.normal(size=(batch_size, 128)))
    x_r=K.placeholder(shape=(batch_size,128))
    for input_t in range(max_no_threads):
        inps = Lambda(lambda x: x[:, input_t])(send_t)
        inpr = Lambda(lambda x: x[:, input_t])(recv_t)
        x_s,h_r,c_r = LSTM(128, recurrent_dropout=0.5,return_sequences=False)(inps,initial_state=[h_s,c_s])
        x_r,h_s,c_s = LSTM(128, recurrent_dropout=0.5,return_sequences=False)(inpr,initial_state=[h_r,c_r])
    x = Dense(64, activation="relu")(x_r)
    #,kernel_regularizer=regularizers.l2(0.003),activity_regularizer=regularizers.l1(0.01))(x)
    #x = Dropout(0.5)(x)
    x = Dense(max_classes, activation="softmax")(x)
    model = Model(inputs=[send_t,recv_t], outputs=x)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




def model_lstm(max_features, max_classes,maxlen, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen,))
    inpc=Input(shape=(4096,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x =Bidirectional(LSTM(64, recurrent_dropout=0.5,return_sequences=True))(x)
    #if attention layer  follows LSTM, return_sequences must be True else False
    #x=AttentionM()(x)
    x=AttentionMV()([x,inpc])
    #x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    #,kernel_regularizer=regularizers.l2(0.003),activity_regularizer=regularizers.l1(0.01))(x)
    #x = Dropout(0.5)(x)
    x = Dense(max_classes, activation="softmax")(x)
    model = Model(inputs=[inp,inpc], outputs=x)
    model.summary()
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def model_lstm_infersent(max_no_sent, max_classes, embed_size):
    inp = Input(shape=(max_no_sent,embed_size,))
    # entire document is  a matrix of  max_no-sent x sentence embedding vector
    x = LSTM(128, recurrent_dropout=0.5,return_sequences=False)(inp)
    #x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    #,kernel_regularizer=regularizers.l2(0.003),activity_regularizer=regularizers.l1(0.01))(x)
    #x = Dropout(0.5)(x)
    x = Dense(max_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_lstm_infersent_attn(max_no_sent, max_classes, embed_size):
    inp = Input(shape=(max_no_sent,embed_size,))
    # entire document is  a matrix of   max_no-sent x sentence embedding vector
    x = LSTM(128, recurrent_dropout=0.5,return_sequences=True)(inp)
    x = AttentionM()(x)
    #x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    #,kernel_regularizer=regularizers.l2(0.003),activity_regularizer=regularizers.l1(0.01))(x)
    #x = Dropout(0.5)(x)
    x = Dense(max_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def model_lstm_infersent_doc(max_classes,embed_size):
    # entire document is avg/max/min pooled into one vector
    inp = Input(shape=(embed_size,))
    x = Dense(64, activation="relu")(inp)
    #,kernel_regularizer=regularizers.l2(0.003),activity_regularizer=regularizers.l1(0.01))(x)
    #x = Dropout(0.5)(x)
    x = Dense(max_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#fotrain=os.path.join("d:/USHUR/PythonProg/DATAREP/reuters","reuters-allcats.csv")
foembed=os.path.join("d:/USHUR/PythonProg/DATAREP/sentiment","glove.dat")
foresults=os.path.join("d:/USHUR/PythonProg/DATAREP/reuters","results.txt")
fr=open(foresults,'a+')
'''
df_L=pd.read_csv(fo)
df_L.info()
df_L=df_L[['rating','text']]
df_L.dropna(inplace=True)
df_L.info()
print(df_L.head())
'''
#df_L['reviews.txt']=df_L.apply(lambda x: str(x['reviews.title'] + x['reviews.text']),axis=1)
#df_L=df_L.iloc[0:100]
df_L=df_L1
df_L['rating']=df_L['rating'].astype('category')
#create a dictionary with values as category names
topic_d = dict(enumerate(df_L['rating'].cat.categories))
#creat a dict where key is cat name and value is index
inverted_topic_d=dict(map(reversed, topic_d.items()))
print(inverted_topic_d)

list_of_topics=np.asarray(df_L['rating'].unique().tolist())
fig1=plt.figure(figsize=(6,8))
axes1=plt.subplot(111)
plt.subplots_adjust(hspace=0.8)
df_L['rating'].value_counts().reindex(list_of_topics).plot(kind="bar",title="Original Frequency of ratings",ax=axes1)
df_L['topic']=df_L['rating'].apply(lambda x:inverted_topic_d[x])
#create  a new column with topic as index value for each class name
plt.show()



# embedding preamble for word embeddings (google, glove, gensim)
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each application.
MAX_SEQUENCE_LENGTH = 200
# This is 200 for GLOVE and 300 for Google
EMBEDDING_DIM = 300
# number fo classes
NUM_CLASSES=len(topic_d)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df_L['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(df_L['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

'''
#only for infersent context vector of title
#MAX_SENT=20
#X=np.zeros((0,MAX_SENT,4096*1))

X,di=InfersentEmbedding(X,MAX_SENT,df_L['text'].values)
print('Shape of data tensor:', X.shape)
'''
#XC=np.zeros((0,4096))
#XC,di=InferDocEmbedding(XC,df_C.values)

foXC=os.path.join("d:/USHUR/PythonProg/DATAREP/sentiment","XC.dat")
#XC.dump(foXC)

XC=np.load(foXC,allow_pickle=True)

Y = pd.get_dummies(df_L['topic'].values)
print('Shape of label tensor:', Y.shape)


X_train, X_test, Y_train, Y_test, XC_train,XC_test= train_test_split(X,Y, XC,test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
print(XC_train.shape,XC_test.shape)
'''
GLOVE_DIR="c:/USHUR/PythonProg/DATAREP/glove.6B"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'),encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False)
embedding_matrix.dump(foembed)
embedding_matrix=np.load(foembed,allow_pickle=True)
# code for glove embedding ends here 
'''


# for google, embedding dim is 300
filename = os.path.join("c:/USHUR/PythonProg/DATAREP",'GoogleNews-vectors-negative300.bin')
w2v_model = KeyedVectors.load_word2vec_format(filename, binary=True)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():
    try:

        w2v_model[[word.lower()]]
        embedding_matrix[i]=w2v_model[[word.lower()]]
    except KeyError:
        try:
            w2v_model[[word.lower()]]
            embedding_matrix[i]=w2v_model[[word.lower()]]
        except KeyError:
            continue

foEM=os.path.join("d:/USHUR/PythonProg/DATAREP/sentiment","EM.dat")
#embedding_matrix.dump(foEM)

embedding_matrix=np.load(foEM,allow_pickle=True)

model=model_lstm(len(word_index)+1,len(topic_d),MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,embedding_matrix)
'''
EMBEDDING_DIM=4096
model=model_lstm_infersent(MAX_SENT,len(topic_d),EMBEDDING_DIM)
'''
epochs = 20
batch_size = 32
#classweight={0:1.7,1:1.0}
history = model.fit([X_train,XC_train], Y_train,epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss', mode='min',patience=30, min_delta=1)])
accr = model.evaluate([X_test,XC_test],Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
#print('Score {:.2f}\n').format(score)
#model.evaluate function predicts the output for the given input and then computes the metrics function specified in the model.compile
#The model.predict just returns back the y_pred

y_pred=model.predict([X_test,XC_test], batch_size=None, verbose=0, steps=None)
print(y_pred.shape,y_pred)
Y_pred=np.argmax(y_pred,axis=1)
print(Y_pred.shape)
print(np.argmax(Y_test.values,axis=1).shape)

accuracy = accuracy_score(np.argmax(Y_test.values,axis=1), Y_pred)
print('Accuracy: %f' % accuracy)
fr.write('Accuracy: %f \n' % accuracy)

cm = confusion_matrix(np.argmax(Y_test.values,axis=1), Y_pred)
print(cm)

plot_confusion_matrix(np.argmax(Y_test.values,axis=1),Y_pred,classes=['negative','neutral','positive'],normalize=True,
                     title='Confusion matrix, with normalization')

plt.show()
exit()
