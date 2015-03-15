
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

import json

with open(filepath+'tfidf_scores.json') as json_data:
    tfidf_scores = json.load(json_data)
    json_data.close()
    
bestwords_tfidf = np.array(sorted(tfidf_scores, key=tfidf_scores.get,reverse=True))


# In[5]:

with open('/Users/DanLo1108/Downloads/modified_text_with_1000_clusters/train_modified_1000.json') as json_data:
    modified_train = json.load(json_data)
    json_data.close()


# In[22]:

modified_train['review'][0]


# In[5]:

#Collect random 80/20 train/test split
import random

train_inds=random.sample(train_data.index.values,20000)
test_inds=[i for i in train_data.index.values if i not in train_inds]

train=train_data.ix[train_inds]
test=train_data.ix[test_inds]


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
    #
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


# In[9]:

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


# In[10]:

def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


# In[19]:

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review,         remove_stopwords=True ))
    
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review,         remove_stopwords=True ))


# In[12]:

def normalize(array):
    return array/sum(array)


# In[22]:

#Runs clustering for different number of clusters

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
denominators = [3.,4.,5.,6.,7.] #denominator of fraction for number of clusters as fraction of length training data
train_AUCs_list=[]
test_AUCs_list=[]
for denom in denominators:
    
    word_vectors = model.syn0
    num_clusters = np.int(word_vectors.shape[0] / denom)

    kmeans_clustering = KMeans(n_clusters = num_clusters)
    idx=kmeans_clustering.fit_predict(word_vectors)
    
    word_centroid_map = dict(zip(model.index2word, idx))
    
    #Creates train centroids
    train_centroids = np.zeros((train['review'].size, num_clusters), dtype='float32')
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
        counter += 1
        
    print 'train reviews complete'
    
    #Creates test centroids
    test_centroids = np.zeros((test['review'].size, num_clusters), dtype='float32')
    counter=0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1
        
    print 'test reviews complete'
        
    cluster_to_word_dict={}

    #Maps cluster to its words
    for cluster in range(num_clusters):
        cluster_to_word_dict[cluster]=[]
        for word in word_centroid_map:
            if word_centroid_map[word] == cluster:
                cluster_to_word_dict[cluster].append(word)
                
    #Finds weights of each cluster based on the average tfidf weights
    #of its words
    cluster_weights_dict={}
    for cluster in cluster_to_word_dict:
        weights=[]
        for word in cluster_to_word_dict[cluster]:
            if word in tfidf_scores:
                weights.append(tfidf_scores[word])
            else:
                weights.append(0)
                
        cluster_weights_dict[cluster] = np.mean(weights)
      
        
    #Finds weights of each centroid based on the average tfidf
    #weights of its clusters
    centroid_weights=[]
    for tc in train_centroids:
        weight=[]
        for clust in tc:
            weight.append(cluster_weights_dict[clust])
        centroid_weights.append(np.mean(weight)) 
    centroid_weights=np.array(centroid_weights)
    centroid_weights=normalize(centroid_weights)
    
    preds=test['sentiment'].tolist()
    
    #Clusters training and testing centroids
    total_centroids=np.concatenate((train_centroids,test_centroids),axis=0)
    centroid_clusters=KMeans(n_clusters=int(np.sqrt(num_clusters)/2))
    centroid_clusters.fit(total_centroids)
    
    train_centroid_clusters = centroid_clusters.predict(train_centroids)
    test_centroid_clusters = centroid_clusters.predict(test_centroids)
    
    print 'centroids clustered'
    
    #Run boosting N times
    N=1 #N=1 - no boosting
    for iteration in range(N):
        
        #This block of code identifies how often each cluster is misclassified
        bad_clusters={}
        for i in range(len(preds)):
            if preds[i] != test['sentiment'].tolist()[i]:
                misclassified_cluster = test_centroid_clusters[i]
                if misclassified_cluster not in bad_clusters:
                    bad_clusters[misclassified_cluster] = 1
                else:
                    bad_clusters[misclassified_cluster] += 1
                
        #Adjusts weights based on frequency of cluster misclassification and
        #how often it appears in training data
        for bc in bad_clusters:
            inds = [ind for ind in range(len(train_centroid_clusters)) if train_centroid_clusters[ind]==bc] 
            cw_inds=centroid_weights[inds]
            centroid_weights[inds]=cw_inds*(1+float(bad_clusters[bc])/len(inds))*5
            centroid_weights=normalize(centroid_weights)
        
        #Fits random forest classifier with weights set to centroid weights
        scores=[]
        test_AUCs=[]
        train_AUCs=[]
        for i in range(3):
            forest = RandomForestClassifier(n_estimators=100)
            forest1 = forest.fit(train_centroids,train["sentiment"],sample_weight=centroid_weights)

            scores.append(forest1.score(test_centroids,test['sentiment']))
            test_AUCs.append(roc_auc_score(test['sentiment'],forest1.predict_proba(test_centroids)[:,1]))
            
        for i in range(3):
            forest = RandomForestClassifier(n_estimators=100)
            forest.fit(train_centroids,train["sentiment"])

            scores.append(forest1.score(train_centroids,train['sentiment']))
            train_AUCs.append(roc_auc_score(train['sentiment'],forest1.predict_proba(train_centroids)[:,1]))

        if (iteration+1) in [1,2,3,4,5,8,10,15,20,25]:
            print (iteration+1), 'num_clusters: ',str(num_clusters),'    acc: ', str(np.mean(scores)), '    AUC: ',str(np.mean(AUCs))     
    
    test_AUCs_list.append(np.mean(test_AUCs))
    train_AUCs_list.append(np.mean(train_AUCs))


# In[ ]:

plot()


# In[41]:

model.most_similar('dog')


# In[61]:

model.most_similar('movi')


# In[ ]:

model.most_similar('good')


# In[ ]:

model.most_similar('bad')

