import fastai
from fastai import *
from fastai.text import *
import torch
import pandas as pd
import numpy as np
from functools import partial
import ast
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings

#downloaded models are storesd in ~/.fastai/models/wt103-fwd
#AWD has 2 files itos.pkl and lstm_fwd.pth
def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
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

        fig, ax = plt.subplots(figsize=(2 * cm.shape[0], 2 * cm.shape[0]))
        im = ax.matshow(cm, cmap=cmap)
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
        # plt.rc('xtick', labelsize=20)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, size=10)
        ax.xaxis.tick_bottom()
        plt.yticks(tick_marks, classes, size=10)
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        # rotation_mode="anchor",size=12)
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

    fotrain=os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK","unumuk.csv")
    folm=os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK","data_lm_tr_export.pkl")
    foclas=os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK","data_clas_tr_export.pkl")
    foftenc=os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK","ft_enc_tr")
    foclfmdl = os.path.join("d:/USHUR/PythonProg/DATAREP/UNUMUK", "clf_model_tr.pkl")
    model_path="d:/USHUR/PythonProg/DATAREP/UNUMUK"
    df_L=pd.read_csv(fotrain)
    df_L.dropna(inplace=True)
    print(df_L.head())
    print(df_L.info())
    df_L['received']=df_L['received'].apply(lambda x: ast.literal_eval(x))
    df_L['sent']=df_L['sent'].apply(lambda x: ast.literal_eval(x))
    MAX_NO_THREADS=5
    def merge_thread(r,s,MAX_NO_THREADS):
        r = (r + MAX_NO_THREADS * [' '])[:MAX_NO_THREADS]
        s = (s + MAX_NO_THREADS * [' '])[:MAX_NO_THREADS]
        '''
        if not r:
            rt=' '
        else:
            rt=r[0]
        if not s:

            st=' '
        else:
            st=s[0]
        '''
        return r[0] + s[0]+r[1]+s[1]+r[2]+s[2]
    df_L['text']=df_L.apply(lambda x:merge_thread(x['received'],x['sent'],MAX_NO_THREADS),axis=1)

    print(df_L.loc[0:4,'text'])

    df_L=df_L.rename(columns={'topic':'label'})
    df_L=df_L[['label','text']]
    df_M=pd.DataFrame()
    df_M['label'] = df_L['label'].astype('category')
    topic_d = dict(enumerate(df_M['label'].cat.categories))
    print(topic_d.values())

    print(df_L.info())

    df_trn, df_test = train_test_split(df_L, stratify=df_L['label'], test_size = 0.1, random_state = 12)
    print("after train test split",df_trn.shape,df_test.shape)
    df_trn, df_val = train_test_split(df_trn, stratify=df_trn['label'], test_size=0.2, random_state=32)
    print("after train val split", df_trn.shape, df_val.shape)
    defaults.cpus=1

    #WINDOWS PRAGMA if you dont have this on windows you get broken pool error

    # Language model data
    data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, test_df=df_test, text_cols='text',label_cols='label',path = "")
    #data_lm = TextLMDataBunch.from_csv(fotrain)
    # Classifier model data
    data_clas = TextClasDataBunch.from_df(path="", train_df=df_trn, valid_df=df_val, test_df=df_test,text_cols='text',label_cols='label',vocab=data_lm.train_ds.vocab,
                                          bs=32)
    #data_clas = TextClasDataBunch.from_csv(fotrain, vocab=data_lm.train_ds.vocab, bs=32)
    data_clas.vocab.itos = data_lm.vocab.itos
    data_clas.vocab.stoi = data_lm.vocab.stoi
    data_lm.save(folm)
    data_clas.save(foclas)
    data_lm = load_data(path='d:/USHUR/PythonProg/DATAREP/UNUMUK',file='data_lm_tr_export.pkl')
    lm = language_model_learner(data_lm, Transformer, drop_mult=0.5)
    # train the learner object with learning rate = 1e-2
    lm.fit_one_cycle(1, 1e-2)
    lm.save_encoder(foftenc)

    #set num_workers=0 to  avoid [Errno 32] Broken pipe
    data_clas = load_data(path='d:/USHUR/PythonProg/DATAREP/UNUMUK', file='data_clas_tr_export.pkl',num_workers=0)
    ltxtc = text_classifier_learner(data_clas, Transformer,drop_mult=0.5)
    ltxtc.load_encoder(foftenc)
    #ltxtc.unfreeze()
    ltxtc.fit_one_cycle(1, 1e-2)
    ltxtc.export(file=foclfmdl)



    # 4. code to predict a batch of test data in its entirety loaded from load_learner
    df_test.reset_index(inplace=True)
    Y = pd.get_dummies(df_test['label'].values)
    print('Shape of label tensor:', Y.shape)
    print(type(Y))
    ltxtsaved = load_learner(path=model_path, file='clf_model_tr.pkl', test=df_test, num_workers=0)
    #ltxtsaved = load_learner(path=model_path, file='clf_model.pkl', num_workers=0)
    #if test_df is given in LMdatbunch and clasd databunch
    preds, targets = ltxtsaved.get_preds(ds_type=DatasetType.Test)
    actual = np.argmax(Y.values, axis=1)
    predictions = np.argmax(preds, axis=1)
    accuracy = accuracy_score(actual, predictions)
    print('Accuracy: %f' % accuracy)
    cm = confusion_matrix(actual, predictions)
    print(cm)
    plot_confusion_matrix(actual, predictions, classes=topic_d.values(), normalize=True,
                          title='Confusion matrix, with normalization')
    plt.show()
    # 4. Code for predicting a batch of  test data in its entirety from load_learner
    exit()

    '''
    # 1. code to predict a single instance
    #returns label, predicted class, probablities
    ltxtsaved = load_learner(path=model_path, file='clf_model.pkl',num_workers=0)
    teststring=df_L.loc[0:0,'text']
    print(type(teststring))
    print(teststring)
    #teststring="fw : quote for name name name : number ) was not collected within the number day holding period .   " \
                    #"expired messages are permanently deleted"
    predlabel,target,preds=ltxtsaved.predict(teststring)
    print(predlabel,target.numpy(),np.argmax(preds.numpy()))
    #code for predicting single test instance



    # 2. code to predict a batch of test data loaded independently
    ltxtsaved = load_learner(path=model_path, file='clf_model.pkl', num_workers=0)
    teststring = df_L.loc[0:5, 'text'].values
    print(type(teststring))
    print(teststring)
    ltxtsaved.data.add_test(teststring)
    # teststring="fw : quote for name name name : number ) was not collected within the number day holding period .   " \
    # "expired messages are permanently deleted"
    preds,y=ltxtsaved.get_preds(ds_type=DatasetType.Test)
    print(np.argmax(preds.numpy(),axis=1),y.numpy())
    # code for predicting  a batch of test data loaded independently
    
    # 3. code to predict a batch of test data loaded from load_learner
    df_test.reset_index(inplace=True)
    Y = pd.get_dummies(df_test['label'].values)
    print(Y[0:50].values.shape)
    print(df_test.loc[0:49,'text'].values.shape)
    print('Shape of label tensor:', Y.shape)
    print(type(Y))
    ltxtsaved=load_learner(path=model_path,file='clf_model.pkl',test=df_test.loc[0:49,'text'],num_workers=0)
    preds, targets = ltxtsaved.get_preds(ds_type=DatasetType.Test)
    actual = np.argmax(Y[0:50].values, axis=1)
    predictions = np.argmax(preds, axis=1)
    accuracy = accuracy_score(actual, predictions)
    print('Accuracy: %f' % accuracy)
    cm = confusion_matrix(actual, predictions)
    print(cm)
    plot_confusion_matrix(actual, predictions, classes=topic_d.values(), normalize=True,
                          title='Confusion matrix, with normalization')
    plt.show()
    #3. Code for predicting a batch of  test data from load_learner
    exit()
    

    #ltxtsaved = load_learner(path=model_path, file='clf_model.pkl',test=df_test, num_workers=0)



#ltxtsaved.data.add_test(["That movie was terrible!",
                 #"I'm a big fan of all movies with Hal 3000."])
#preds,y=ltxtsaved.get_preds(ds_type=DatasetType.Test)
#predict row by row from a dataframe
#test_pred['label'] = test_pred['text'].apply(lambda row: str(ltxtsaved.predict(row)[0]))

#df_test.reset_index(inplace=True)
#sample=df_test.loc[0,'text']
#print(type(sample))
preds,target=ltxtsaved.get_preds(ds_type=DatasetType.Test)
#print(preds)
#print(targets)

print(predictions)

print(np.argmax(Y.values,axis=1))

#pd.crosstab(predictions, targets
#print(pd)
#accuracy = accuracy_score(np.argmax(preds,axis=1), targets)

#fr.write('Accuracy: %f \n' % accuracy)


exit()


preds, targets = ltxtsaved.get_preds(ds_type=DatasetType.Test, with_loss=True)
predictions = np.argmax(preds, axis=1)
pd.crosstab(predictions, targets)
print(pd)
accuracy = accuracy_score(np.argmax(preds, axis=1), targets)
print('Accuracy: %f' % accuracy)
# fr.write('Accuracy: %f \n' % accuracy)

cm = confusion_matrix(np.argmax(preds, axis=1), targets)
print(cm)

plot_confusion_matrix(np.argmax(preds, axis=1), targets, classes=topic_d.values(), normalize=True,
                      title='Confusion matrix, with normalization')
plt.show()

exit()
learn = language_model_learner(data_lm, pretrained_fnames=[name_of_the_pth_file, name_of_the_pkl_file], drop_mult=0.3)
#get predictions on your own test set
preds, y, losses = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)
y = torch.argmax(preds, dim=1)
#another approach
preds = []
for i in range(0,test.data_size):
p = learn.predict(data.test_ds.x[i])
preds.append(str(p[0]))
'''