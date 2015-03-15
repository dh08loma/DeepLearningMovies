
# coding: utf-8

# In[13]:

import pandas as pd
import numpy as np
import nltk
import string
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk.data
from nltk import stem
from sklearn import cross_validation
from pylab import *


# In[14]:

filepath = '/Users/DanLo1108/Documents/Grad School Files/Advanced ML/Final Project/'


# In[15]:

train = pd.read_table(filepath + 'labeledTrainData.tsv')
test = pd.read_table(filepath + 'testData.tsv')


# In[4]:

# The data is pretty simple. It consists of a review ID (each of these are unique,
# so I'm pretty sure we can disregard this column), sentiment (0 is for a bad review,
# 1 is for a good review), and the review (block of text)

train.head()


# In[5]:

# Is the training data balanced?
# Yes - it is exactly balanced between classes

len(train[train.sentiment == 1])/float(len(train))


# In[6]:

# How long are the reviews?

# Roughly no difference between length of positive reviews and 
# length of negative reviews

word_lengths = [len(st) for st in train.review]
word_lengths_pos = [len(st) for st in train[train.sentiment == 1].review]
word_lengths_neg = [len(st) for st in train[train.sentiment == 0].review]


print 'Mean num words total: ',np.mean(word_lengths)
print 'Mean num words positive: ',np.mean(word_lengths_pos)
print 'Mean num words negative: ',np.mean(word_lengths_neg)


##### Data cleaning

# In[7]:

# This function cleans up a review into something more usable.
# Input: raw review
# Output: cleaned review

def clean_review(x,stem):
    
    #Remove HTML
    raw_review = BeautifulSoup(x.review.decode("utf8")).get_text() 
    
    #Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    
    #Converts all letters to lowercase
    rev_lower = letters_only.lower()
    
    #Tokenizes review using nltk
    rev_token = nltk.word_tokenize(rev_lower)
    
    #Removes punctuation and stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    rev_sw = [w for w in rev_token if w not in stopwords]
    
    punc = string.punctuation
    rev_punc = [w for w in rev_sw if w not in punc]
    
    #Apply stemmer
    rev = [stem.stem(r) for r in rev_punc]
    
    return rev


####### Let's compare different stemmers to find best one:

# In[8]:

from nltk import stem
x=train.ix[1] #arbitrary review observation

#3 types of stemmers
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
snowball = stem.snowball.EnglishStemmer()

port=clean_review(x,porter)
lanc=clean_review(x,lancaster)
snow=clean_review(x,snowball)


# In[9]:

#Gets indices of differing stemmer word interpretations
diff=[i for i in range(len(lanc)) if lanc[i] != snow[i] or lanc[i] != port[i] or snow[i] != port[i]]


# In[10]:

# Results: Lancaster works better for words which end in 'y',
# however it strips much more of the word than porter or snowball.

# I would favor snowball or porter over lancaster.
# Snowball is a slightly improved version of porter
# so that would be my preferance.

diff=[i for i in range(len(lanc)) if lanc[i] != snow[i]]
print 'Differing words for stemmers\n'
print 'lancaster|porter|snowball\n'
for d in diff:
    print lanc[d],',', port[d],',', snow[d]


# In[11]:

# Convert all reviews to cleaned version: **Takes ~5 minutes

#Train
train['review'] = train.apply(lambda x: clean_review(x,snowball), axis=1)
#Test
test['review'] = test.apply(lambda x: clean_review(x,snowball), axis=1)


# In[12]:

train['review_clean']=train['review'].apply(lambda x: string.join(x))

test['review_clean']=test['review'].apply(lambda x: string.join(x))


##### Example Bag of Words

# In[12]:

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word", max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()


# In[31]:

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab


# In[13]:


# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )


# In[14]:

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)


# In[16]:

forest.score(test_data_features)


#### Let's find the optimal number of words in vectorizer

# In[45]:

N_features=np.arange(1000,11000,1000)
accuracies=[]

for n in N_features:
    
    accs=[]
    
    #Create features
    vectorizer = CountVectorizer(analyzer = "word", max_features = n) 
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
        
    for i in range(3):

        X_train,X_test,y_train,y_test=cross_validation.train_test_split(train_data_features,train.sentiment,test_size=.2)
        
        
        forest = RandomForestClassifier(n_estimators = 100) 
        forest = forest.fit(X_train, y_train)
        
        accs.append(forest.score(X_test,y_test))
        
    accuracies.append(np.mean(accs))
    print n


# In[56]:

plot(N_features,accuracies,label='3 fold random cross-validation, 80/20 split')
title('Bag of Words accuracy varying number of features',fontsize=20)
xlabel('Number of words as features',fontsize=16)
ylabel('Accuracy',fontsize=16)
legend()
show()


#### Plot the learning curve for BOW

# In[28]:

import random
percent_train = [.50,.60,.70,.80,.90,1.0]
accuracies=[]

for perc in percent_train:
    
    accs=[]
    
    #Create features
    vectorizer = CountVectorizer(analyzer = "word", max_features = 9000) 
    rand = random.sample(train.index.values,int(len(train)*perc))
    train1 = train.ix[rand]
    train_data_features = vectorizer.fit_transform(train1.review_clean)
    train_data_features = train_data_features.toarray()
        
    for i in range(3):

        X_train,X_test,y_train,y_test=cross_validation.train_test_split(train_data_features,train1.sentiment,test_size=.2)
        
        
        forest = RandomForestClassifier(n_estimators = 100) 
        forest = forest.fit(X_train, y_train)
        
        accs.append(forest.score(X_test,y_test))
        
    accuracies.append(np.mean(accs))
    print perc


# In[34]:

np.array(percent_train)*25000


# In[35]:

plot(np.array(percent_train)*25000,accuracies,label='3 fold random cross-validation, 80/20 split')
title('Bag of Words accuracy varying size of data',fontsize=20)
xlabel('Number of training+test observations',fontsize=16)
ylabel('Accuracy',fontsize=16)
legend()
show()


#### Find optimal number of trees in forest

# In[36]:

import random
num_trees = [5,10,50,100,200,500]
accuracies=[]

for t in num_trees:

    accs=[]
    
    #Create features
    vectorizer = CountVectorizer(analyzer = "word", max_features = 9000) 
    train1 = train
    train_data_features = vectorizer.fit_transform(train1.review_clean)
    train_data_features = train_data_features.toarray()
        
    for i in range(3):

        X_train,X_test,y_train,y_test=cross_validation.train_test_split(train_data_features,train1.sentiment,test_size=.2)
        
        
        forest = RandomForestClassifier(n_estimators = t) 
        forest = forest.fit(X_train, y_train)
        
        accs.append(forest.score(X_test,y_test))
        
    accuracies.append(np.mean(accs))
    print t


# In[38]:

plot(num_trees,accuracies,label='3 fold random cross-validation, 80/20 split')
title('Bag of Words accuracy varying number of trees',fontsize=20)
xlabel('Number of trees',fontsize=16)
ylabel('Accuracy',fontsize=16)
show()


# In[ ]:



