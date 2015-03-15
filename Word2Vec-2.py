
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import nltk
from nltk import stem
import string
from nltk.stem.snowball import SnowballStemmer
import math


# In[2]:

filepath = '/Users/DanLo1108/Documents/Grad School Files/Advanced ML/Final Project/'


# In[3]:

#Import data

train_data = pd.read_csv(filepath + "labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
test_data = pd.read_csv(filepath + "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv(filepath + "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )


# In[4]:

train=train_data
test=test_data


# In[5]:

#Collect random 80/20 train/test split
import random

train_inds=random.sample(train_data.index.values,20000)
test_inds=[i for i in train_data.index.values if i not in train_inds]

train=train_data.ix[train_inds]
test=train_data.ix[test_inds]


# In[56]:

#positive vs negative reviews

pos = train[train.sentiment==1]
neg = train[train.sentiment==0]


# In[6]:

# Import various modules for string cleaning

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

stemmer=SnowballStemmer('english')

def review_to_wordlist( review, remove_stopwords=False, stem=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text() 
    
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        
    #stemming
    words = [stemmer.stem(w) for w in words]
    
    #
    # 5. Return a list of words
    return(words)


# In[41]:

# Convert pos and neg reviews to cleaned version: **Takes ~5 minutes

#Positive
pos['review'] = pos.apply(lambda x: review_to_wordlist(x.review,remove_stopwords=True), axis=1)
#Negative
neg['review'] = neg.apply(lambda x: review_to_wordlist(x.review,remove_stopwords=True), axis=1)


# In[91]:

#Gets cleaned reviews for all training data

clean_train_reviews=train.apply(lambda x: review_to_wordlist(x.review,remove_stopwords=True), axis=1)


#### Create model

# In[7]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.decode("utf8").strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, stem=False ))
    
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


# In[8]:

#Create sentences to feed into model

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


# In[26]:

#Creates word2Vec model
#Loops through several minimum word and num_features parameters
#to find best model
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

#num_features=[100,200,300,400,500]
min_word_counts=[20,30,40,50,60]
#for nf in num_features:
for mwc in min_word_counts:
    # Set values for various parameters
    num_features = 300     # Word vector dimensionality                      
    min_word_count = mwc   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model 
    from gensim.models import word2vec
    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers,                 size=num_features, min_count = min_word_count,                 window = context, sample = downsampling)


    #Names and saves model for later
    #model_name = str(nf)+"features_40minwords_10context"
    model_name = "300features_"+str(mwc)+"minwords_10context"
    print model_name
    model.save(model_name)


# In[14]:

from gensim.models import word2vec
model=word2vec.Word2Vec.load("100features_40minwords_10context")


#### Using tfidf to weigh words

# In[ ]:

#Dictionary which contains the number of documents each word appears in

n_containing={}
for word in words:
    n_containing[word] = sum(1 for review in clean_train_reviews if word in review)


# In[ ]:

#TFIDF functions

def tf(word,review):
    count=Counter(review)
    return float(count[word])/len(review)

def idf(word,review_list,n_containing):
    return math.log(len(review_list)/(1+n_containing[word]))

def tfidf(word,review,review_list,n_containing):
    return tf(word,review)*idf(word,review_list,n_containing)


# In[ ]:

#Gets tfidf score for every word in every document, for pos and neg,
#Then finds the average tfidf for each word in positive or negative class

# ***Takes a LONG time - I saved sample json files which I'll send along

pos_tfidf={}
neg_tfidf={}
words=model.index2word
for word in words:
    for review in pos.review:
        word_tfidf=tfidf(word,review,clean_train_reviews,n_containing)
        if word in pos_tfidf:
            pos_tfidf[word].append(word_tfidf)
        else:
            pos_tfidf[word]=[word_tfidf]
    pos_tfidf[word]=np.mean(pos_tfidf[word])
            
    for review in neg.review:
        word_tfidf=tfidf(word,review,clean_train_reviews,n_containing)
        if word in neg_tfidf:
            neg_tfidf[word].append(word_tfidf)
        else:
            neg_tfidf[word]=[word_tfidf]
    neg_tfidf[word]=np.mean(neg_tfidf[word])
            


# In[421]:

#Function which finds tfidf score

def get_word_tfidf(word,dic):
    if word in dic:
        return dic[word]
    else:
        return 0


# In[422]:

#Gets tfidf score for each word

#Score is defined as the absolute difference between pos and neg,
#which gives higher weight to words more heavily associated
#with one side

words=model.index2word
tfidf_scores={}

for word in words:
    pos_freq=get_word_tfidf(word,pos_tfidf)
    neg_freq=get_word_tfidf(word,neg_tfidf)
    tf_diff=abs(pos_freq-neg_freq)
    
    tfidf_scores[word]=tf_diff


# In[54]:

#Imports TF-IDF dictionary

import json

with open(filepath+'tfidf_scores.json') as json_data:
    tfidf_scores = json.load(json_data)
    json_data.close()
    
#Best words are sorted by TF-IDF rating
bestwords_tfidf = np.array(sorted(tfidf_scores, key=tfidf_scores.get,reverse=True))


#### Getting best words using Chi Squared

# In[61]:

#Gets positive and negative words

pos_words=[]
neg_words=[]
for i in range(len(pos)):
    p=pos.index.values[i]
    pos_words+=pos.ix[p].review
    
for i in range(len(neg)):
    n=neg.index.values[i]
    neg_words+=neg.ix[n].review


# In[62]:

#Use chisq to find top words

from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures

word_fd = FreqDist(pos_words+neg_words)
label_word_fd = ConditionalFreqDist()

label_word_fd['pos'] = FreqDist(pos_words)
label_word_fd['neg'] = FreqDist(neg_words)


pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count


# In[63]:

word_scores = {}
 
for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
        (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
        (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score


best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:100]
bestwords = set([w for w, s in best])
bestwords_chisq = sorted(word_scores, key=word_scores.get,reverse=True)


# In[15]:

#This function creates a feature vector without using word weights

def makeFeatureVec(words, model, num_features):

    featureVec = np.zeros((num_features,),dtype="float32")
    
    nwords = 0.
    
 
    index2word_set = set(model.index2word)
    
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


#Creates feature vector using word weights
def makeFeatureVec1(words, model, tfidf_scores, num_features):
  
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    
    
    weight_sum=0.0
    weight_sum+=np.sum(tfidf_scores[word] for word in words if word in index2word_set and word in tfidf_scores)
        
    for word in words:
        if word in index2word_set and word in tfidf_scores: 
            weighted_score=(model[word]*tfidf_scores[word])/weight_sum
            #print len(featureVec)
            #print len(weighted_score)
            nwords = nwords + 1.
            #featureVec = np.add(featureVec,model[word]/weight_sum)
            featureVec = np.add(featureVec,weighted_score)
     
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


#

#weights = indicator for if word weights should be used
#tf
def getAvgFeatureVecs(reviews, model, tfidf_score, num_features,weights):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 

    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
        
       # Call the function (defined above) that makes average feature vectors
        if weights==True:
            reviewFeatureVecs[counter] = makeFeatureVec1(review, model, tfidf_score,                num_features)
        else:
            reviewFeatureVecs[counter] = makeFeatureVec(review, model,                 num_features)

    return reviewFeatureVecs


# In[17]:

#Gets training and test feature vectors without weights

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review,         remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, tfidf_scores, num_features,weights=False )

#print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review,         remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, tfidf_scores, num_features,weights=False )


# In[21]:

#Gets training and test feature vectors with weights

clean_train_reviews1 = []
for review in train["review"]:
    clean_train_reviews1.append( review_to_wordlist( review,         remove_stopwords=True ))

trainDataVecs1 = getAvgFeatureVecs( clean_train_reviews1, model, tfidf_scores, num_features,weights=True )

#print "Creating average feature vecs for test reviews"
clean_test_reviews1 = []
for review in test["review"]:
    clean_test_reviews1.append( review_to_wordlist( review,         remove_stopwords=True ))

testDataVecs1 = getAvgFeatureVecs( clean_test_reviews1, model, tfidf_scores, num_features,weights=True )


# In[64]:

# Import various modules for string cleaning

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

stemmer=SnowballStemmer('english')

def review_to_wordlist2( review, bestwords, remove_stopwords=False, stem=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text() 
    
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        
    #stemming
    words = [stemmer.stem(w) for w in words]
    
    #words = [words]
    
    words = [words[i] for i in range(len(words)) if words[i] in bestwords]
    
    #
    # 5. Return a list of words
    return(words)


# In[45]:

#3 fold cross validation for varying number of features in W2V model

num_features=[100,200,300,400,500]
min_word_counts=[20,30,40,50,60]
AUC_list_nf=[]
#AUC_list_mwc=[]
for nf in num_features:
#for mwc in min_word_counts:
    model_name=str(nf)+"features_40minwords_10context"
    #model_name="300features_"+str(mwc)+"minwords_10context"

    model=word2vec.Word2Vec.load(model_name)

    trainDataVecs1 = getAvgFeatureVecs( clean_train_reviews1, model, tfidf_scores, nf,weights=True)
    testDataVecs1 = getAvgFeatureVecs( clean_test_reviews1, model, tfidf_scores, nf,weights=True )
    
    #trainDataVecs1 = getAvgFeatureVecs( clean_train_reviews1, model, tfidf_scores, 300,weights=True)
    #testDataVecs1 = getAvgFeatureVecs( clean_test_reviews1, model, tfidf_scores, 300,weights=True )
    
    AUCs=[]
    for i in range(3):
        from sklearn.ensemble import RandomForestClassifier
        forest1 = RandomForestClassifier( n_estimators = 100 )
        forest1.fit( trainDataVecs1, train["sentiment"] )

        # Test & extract results 
        from sklearn.metrics import roc_auc_score
        result1 = forest1.predict( testDataVecs1 )
        score1 = forest1.score(testDataVecs1,test['sentiment'])
        AUC1=roc_auc_score(test['sentiment'],forest1.predict_proba(testDataVecs1)[:,1])
        AUCs.append(AUC1)
    
    AUC_list_nf.append(np.mean(AUCs))
    #AUC_list_mwc.append(np.mean(AUCs))
    print nf


# In[50]:

from pylab import *
plot(num_features,AUC_list_nf)
xlabel('Number of features in Word2Vec',fontsize=16)
ylabel('AUC',fontsize=16)
xlim([50,550])
ylim([.920,.930])
title('5-fold CV: AUC vs Number of Features in Word2Vec', fontsize=20)
show()


# In[52]:

from pylab import *
plot(min_word_counts,AUC_list_mwc)
xlabel('Minimum word count threshold in Word2Vec',fontsize=16)
ylabel('AUC',fontsize=16)
xlim([15,65])
ylim([.915,.930])
title('5-fold CV: AUC vs Minimum Word Count Threshold in Word2Vec', fontsize=17)
show()


# In[53]:

#Num words with TF-IDF and Chi Squared Cross Validation
#Gets training and test feature vectors with weights

num_words_list=[2000,4000,5000,6000,8000,10000]

chisq_accs=[]
chisq_aucs=[]
tfidf_accs=[]
tfidf_aucs=[]

for num_words in num_words_list:
    bestwords_list=[bestwords_chisq[:num_words],bestwords_tfidf[:num_words]]
    c=0
    for bestwords in bestwords_list:
        
        clean_train_reviews2 = []
        for review in train["review"]:
            clean_train_reviews2.append( review_to_wordlist2( review, bestwords,                 remove_stopwords=True ))

        trainDataVecs2 = getAvgFeatureVecs(clean_train_reviews2, model, tfidf_scores, num_features,weights=True )

        #print "Creating average feature vecs for test reviews"
        clean_test_reviews2 = []
        for review in test["review"]:
            clean_test_reviews2.append( review_to_wordlist2( review, bestwords,                 remove_stopwords=True ))

        testDataVecs2 = getAvgFeatureVecs(clean_test_reviews2, model, tfidf_scores, num_features,weights=True )

        inds_train = np.array([i for i in range(len(trainDataVecs2)) if True not in np.isnan(trainDataVecs2[i])])
        inds_test = np.array([i for i in range(len(testDataVecs2)) if True not in np.isnan(testDataVecs2[i])])
        
        trainDataVecs2=trainDataVecs2[inds_train]
        testDataVecs2=testDataVecs2[inds_test]
        
        y_train=train.sentiment[train.index.values[inds_train]]
        y_test=test.sentiment[test.index.values[inds_test]]

        #Fits and scores model with weights

        from sklearn.ensemble import RandomForestClassifier
        forest2 = RandomForestClassifier( n_estimators = 100 )

        #print "Fitting a random forest to labeled training data..."
        forest2 = forest2.fit( trainDataVecs2, y_train )

        # Test & extract results 
        from sklearn.metrics import roc_auc_score
        result2 = forest2.predict( testDataVecs2 )
        score2 = forest2.score(testDataVecs2,y_test)
        AUC2=roc_auc_score(y_test,forest2.predict_proba(testDataVecs2)[:,1])
        
        if c==0:
            print 'chi_sq: num_words: ',str(num_words),'    acc: ', str(score2), '    AUC: ',str(AUC2)
            print ' '
            chisq_accs.append(score2)
            chisq_aucs.append(AUC2)
            
        if c==1:
            print 'tfidf: num_words: ',str(num_words),'    acc: ', str(score2), '    AUC: ',str(AUC2)
            print ' '
            tfidf_accs.append(score2)
            tfidf_aucs.append(AUC2)
            
        c+=1
            
            


# In[305]:

#Fits and scores model without weights

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a random forest to labeled training data..."
forest = forest.fit( trainDataVecs, train["sentiment"] )

# Test & extract results 
from sklearn.metrics import roc_auc_score
result = forest.predict( testDataVecs )
score = forest.score(testDataVecs,test['sentiment'])
AUC=roc_auc_score(test['sentiment'],forest.predict_proba(testDataVecs)[:,1])


# In[ ]:



