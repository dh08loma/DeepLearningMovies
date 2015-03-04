
# coding: utf-8

# In[95]:

import pandas as pd
import numpy as np
import nltk
from nltk import stem
import string
from nltk.stem.snowball import SnowballStemmer
import math


# In[4]:

filepath = '/Users/DanLo1108/Documents/Grad School Files/Advanced ML/Final Project/'


# In[361]:

#Import data

train_data = pd.read_csv(filepath + "labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
test_data = pd.read_csv(filepath + "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv(filepath + "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )


# In[362]:

#Collect random 80/20 train/test split

train_inds=random.sample(train_data.index.values,20000)
test_inds=[i for i in train_data.index.values if i not in train_inds]

train=train_data.ix[train_inds]
test=train_data.ix[test_inds]


# In[363]:

#positive vs negative reviews

pos = train[train.sentiment==1]
neg = train[train.sentiment==0]


# In[364]:

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
    review_text = BeautifulSoup(review.decode("utf8")).get_text() 
    
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


# In[365]:

# Convert pos and neg reviews to cleaned version: **Takes ~5 minutes

#Positive
pos['review'] = pos.apply(lambda x: review_to_wordlist(x.review,remove_stopwords=True), axis=1)
#Negative
neg['review'] = neg.apply(lambda x: review_to_wordlist(x.review,remove_stopwords=True), axis=1)


# In[366]:

#Gets cleaned reviews for all training data

clean_train_reviews=train.apply(lambda x: review_to_wordlist(x.review,remove_stopwords=True), axis=1)


#### Create model

# In[187]:


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
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


# In[188]:

#Create sentences to feed into model

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


# In[189]:

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


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


# In[477]:

#This function creates a feature vector without using word weights

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    
    index2word_set = set(model.index2word)
    #
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
def makeFeatureVec1(words, model, tfidf_scores, num_features,):
  
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    
    weight_sum=0.0
    weight_sum+=np.sum(tfidf_scores[word] for word in words if word in index2word_set)
        
    for word in words:
        if word in index2word_set: 
            weighted_score=(model[word]*tfidf_scores[word])/weight_sum
            nwords = nwords + 1.
            #featureVec = np.add(featureVec,model[word]/weight_sum)
            featureVec = np.add(featureVec,weighted_score)
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


#

#weights = indicator for if word weights should be used
#tf
def getAvgFeatureVecs(reviews, model, tfidf_score, num_features,weights):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       #if counter%1000. == 0.:
       #    print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
        if weights==True:
            reviewFeatureVecs[counter] = makeFeatureVec1(review, model, tfidf_score,                num_features)
        else:
            reviewFeatureVecs[counter] = makeFeatureVec(review, model,                 num_features)
       #
       # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


# In[478]:

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


# In[479]:

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


# In[468]:

#To use if cleaned train reviews are already defined (quicker)

trainDataVecs1 = getAvgFeatureVecs( clean_train_reviews1, model, tfidf_scores, num_features,word2vec=False )
testDataVecs1 = getAvgFeatureVecs( clean_test_reviews1, model, tfidf_scores, num_features,word2vec=False )


# In[480]:

#Fits and scores model without weights

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a random forest to labeled training data..."
forest = forest.fit( trainDataVecs, train["sentiment"] )

# Test & extract results 
result = forest.predict( testDataVecs )
score = forest.score(testDataVecs,test['sentiment'])


# In[481]:

#Fits and scores model with weights

from sklearn.ensemble import RandomForestClassifier
forest1 = RandomForestClassifier( n_estimators = 100 )

print "Fitting a random forest to labeled training data..."
forest1 = forest1.fit( trainDataVecs1, train["sentiment"] )

# Test & extract results 
result1 = forest1.predict( testDataVecs1 )
score1 = forest1.score(testDataVecs1,test['sentiment'])


# In[482]:

#Score of non-weighted
score


# In[484]:

#Score of weighted
score1


# In[ ]:



