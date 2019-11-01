from nltk.tokenize import word_tokenize
import numpy as np
import nltk

# Extractor Settings ###############################################################################################
# User defined.
CUSTOM_TAGS=['ROLE','LOC','CID','QTY','POLICY','TIME','PLAN','DES']

# Interesting standard tags.
NLTK_TAG_LIST=['NNP','CD','NNS','VBG']

# Total tag list.
TAG_LIST= CUSTOM_TAGS + NLTK_TAG_LIST

# Tags to be intepreted as values.
VALUE_LIST=['NNP','CD','TIME','PLAN','DES']

# List of context words.
CONTEXT=['life',
         'death',
         'STD',
         'LTD',
        'injury',
        'sickness']

# Types of relationships to be extracted.
RELATIONSHIP_LIST=[#Allow reverse
                   ('ROLE','NNP'),
                   ('NNP','ROLE'),
                   
                   ('LOC','NNP'),
                   ('NNP','LOC'),
    
                   ('CID','CD'),
                   ('CD','CID'),
    
                   ('QTY','CD'),
                   ('CD','QTY'),
                    
                   ('QTY','DES'),
                   ('DES','QTY'),
                   
                   ('VBG','TIME'),
                   ('TIME','VBG'),
    
                   ('POLICY','PLAN'),
                   ('PLAN','POLICY')]


# Helper functions #################################################################################################

# Cosine function
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Extract a dicitonary of context:(key,value pairs from the hybrid_context_word_tag list)
def extract(hybrid_context_word_tag):
    context_dict={}
    #For brevity
    hcwt=hybrid_context_word_tag
    # window size is 2
    windowS=list(zip(hcwt[:-1],hcwt[1:]))
    
    #Try forward capture.
    i=0
    while(i<len(windowS)):
        window=windowS[i]
        
        #Relationship match and context match.
        if((window[0][2],window[1][2]) in RELATIONSHIP_LIST and window[0][0]==window[1][0]):
            #Check if context dicitonary exists
            if(window[0][0] in context_dict):
                #Save key value pair to the context dictionary.
                if(window[0][2] in VALUE_LIST):
                    context_dict[window[0][0]][window[1][1]]=window[0][1]
                else:
                    context_dict[window[0][0]][window[0][1]]=window[1][1]
            else:
                #Create a context dicitonary.
                if(window[0][2] in VALUE_LIST):
                    context_dict[window[0][0]]={window[1][1]:window[0][1]}
                else:
                    context_dict[window[0][0]]={window[0][1]:window[1][1]}
                
            #Delete the key value pair from consideration.
          #  print('Delteting: ',windowS[i])
            del windowS[i]
            if(i<len(windowS)):
              #  print('Deleting: ',windowS[i])
                del windowS[i]
        else:
        #Not a relationship match
            i+=1
    
    # Gather unclaimed values
    for (word,_) in windowS:
        if(word[2] in VALUE_LIST):
             #Check if context dicitonary exists
            if(word[0] in context_dict):
                if('Unclaimed_values' in context_dict[word[0]]):
                    context_dict[word[0]]['Unclaimed_values'].append(word[1])
                else:
                    context_dict[word[0]]['Unclaimed_values']=[word[1]]
            else:        
                #Create a context dicitonary.
                context_dict[word[0]]={'Unclaimed':[word[1]]}
    
    return context_dict
    

#Returns a combo of custom and important NLTK tags for words, filters out irrelevant words
def get_hybrid_tagging(custom_tagging,nltk_tagging):
    hybrid_tagging=[]
    for (cust,nlt) in zip(custom_tagging,nltk_tagging):
        if cust!='IR':
            hybrid_tagging.append(cust)
        elif nlt in NLTK_TAG_LIST:
            hybrid_tagging.append(nlt)
        else:
            hybrid_tagging.append('IR')
        
    return hybrid_tagging

#Assumes IR words have been filtered.
def combine_intermediaries(context_word_tag):
    prev_tag=''
    prev_word=''
    prev_context=''
    temp=''
    new_list=[]
    for cwt in context_word_tag:
        (context,word,tag)=cwt
        if(tag[2:] in CUSTOM_TAGS):
            if(tag[0]=='B'):
                #Save buffer to list
                if(temp!=''):
                    new_list.append((prev_context,temp,prev_tag[2:]))
                temp=word
            else:
                if(tag[2:]==prev_tag[2:]and context==prev_context):
                    temp+=' '+word
                else:
                    #Save buffer to list
                    if(temp!=''):
                        new_list.append((prev_context,temp,prev_tag[2:]))
                    temp=word
        else:
            #Save buffer to list
            if(temp!=''):
                new_list.append((prev_context,temp,prev_tag[2:]))
                temp=''
            if(tag=='NNS' and prev_tag=='CD' and prev_context==context):
                word=new_list[-1][1]+' '+word
                del new_list[-1]
                tag=prev_tag
            elif(tag=='NNP' and prev_tag=='NNP' and prev_context==context):
                word=new_list[-1][1]+' '+word
                del new_list[-1]
                tag=prev_tag
            # Save tag to list
            new_list.append((context,word,tag))
        
        
        prev_tag=tag
        prev_word=word
        prev_context=context
    
    #Save buffer to list
    if(temp!=''):
        new_list.append((prev_context,temp,prev_tag[2:]))
        temp=''
    #Return new list.
    return new_list 

def get_context(tokens,vectorizer):
    context='Untitled_context'
    context_list=[]
    for word in tokens:
        for x in CONTEXT:
            if(cosine(vectorizer(word),vectorizer(x))>0.8):
                context=x
                break;
        context_list.append(context)    
    return context_list


# Main funciton ####################################################################################################

def extractor(ts,custom_tagging,vectorizer):
    # Tokeinze word
    tokenized_text = word_tokenize(ts)
    
    # Get context of words.
    context=get_context(tokenized_text,vectorizer)
    
    # Get standard tagging
    nltk_tagging = [y for (x,y) in nltk.pos_tag(tokenized_text)]

    #Create hybrid tagging (also filters irrelevant words)
    hybrid_tagging= get_hybrid_tagging(custom_tagging,nltk_tagging)
    
    # Zip context, words and tagging
    hybrid_context_word_tag=list(zip(context,tokenized_text,hybrid_tagging))

    #Filter irrelevant tags
    hybrid_context_word_tag=[(context,word,tag) for (context,word,tag) in hybrid_context_word_tag if tag!='IR']
   
    #Concat B-I tags to get 'super' custom tags and drop prfixes.
    #The only 'super' nltk tags are CD-NNS and NNP-NNP
    hybrid_context_word_tag= combine_intermediaries(hybrid_context_word_tag)
    
    return extract(hybrid_context_word_tag)

