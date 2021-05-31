#!/usr/bin/env python
# coding: utf-8

# ### Import Statement

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import matplotlib
from matplotlib import pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# ### Load Data

# In[2]:


data = pd.read_csv("books.csv", sep=';')


# In[3]:


data


# ### Number of Example in each class

# In[4]:


# extracting the number of examples of each class
EAP_len = data[data['author'] == 'EAP'].shape[0]
SACD_len = data[data['author'] == 'SACD'].shape[0]
AC_len = data[data['author'] == 'AC'].shape[0]


# In[5]:


# bar plot of the 3 classes
plt.figure(figsize=(10,7))
plt.bar(10,EAP_len,3, label="Edgar Allan Poe")
plt.bar(15,SACD_len,3, label="Sir Arthur Conan Doyle")
plt.bar(20,AC_len,3, label="Agatha Christie")
plt.legend()
plt.ylabel('Number of examples')
plt.title('Proportion of examples')
plt.show()


# ## Feature Engineering

# ### Removing Punctuation

# In[6]:


def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


# In[7]:


data['text'] = data['text'].apply(remove_punctuation)
data.head(10)


# ### Removing Stopwords

# In[8]:


# extracting the stopwords from nltk library
sw = stopwords.words('english')
# displaying the stopwords
np.array(sw)


# In[9]:


print("Number of stopwords: ", len(sw))


# In[10]:


def stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)


# In[11]:


data['text'] = data['text'].apply(stopwords)
data.head(10)


# ### Top Words Before Stemming

# In[12]:


vec = CountVectorizer().fit(data['text'])
bag_of_words = vec.transform(data['text'])
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
words_freq[:10]

from pandas import DataFrame
Top_10_words = DataFrame (words_freq[:10],columns=['word', 'total'])
Top_10_words


# In[13]:


import numpy as np                                                               
import matplotlib.pyplot as plt

#top=[('a',1.875),('c',1.125),('d',0.5)]
words_freq[:10]

labels, ys = zip(*words_freq[:10])
xs = np.arange(len(labels)) 
width = 1

plt.figure(figsize=(10,5))
plt.bar(xs, ys, width=0.7, align='center', color='black')

plt.xticks(xs, labels) #Replace default x-ticks with xs, then replace xs with labels
plt.yticks(ys)


# In[14]:


# Python program to generate WordCloud 

# Text of all words in column bloom

text = " ".join(word for word in data['text'])
print ("There are {} words in the combination of all cells in all books.".format(len(text)))

# Generate a word cloud image

wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)

# Display the generated image:
# the matplotlib way:

plt.axis("off")
plt.figure( figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()


# ### Stemming Operations

# In[15]:


# create an object of stemming function
stemmer = SnowballStemmer("english")

def stemming(text):    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 


# In[16]:


data['text'] = data['text'].apply(stemming)
data.head(10)


# ### Top Words After Stemming

# In[17]:


vec = CountVectorizer().fit(data['text'])
bag_of_words = vec.transform(data['text'])
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
words_freq[:10]

from pandas import DataFrame
Top_10_words = DataFrame (words_freq[:10],columns=['word', 'total'])
Top_10_words


# In[18]:


import numpy as np                                                               
import matplotlib.pyplot as plt

#top=[('a',1.875),('c',1.125),('d',0.5)]
words_freq[:10]

labels, ys = zip(*words_freq[:10])
xs = np.arange(len(labels)) 
width = 1

plt.figure(figsize=(10,5))
plt.bar(xs, ys, width=0.7, align='center', color='black')

plt.xticks(xs, labels) #Replace default x-ticks with xs, then replace xs with labels
plt.yticks(ys)


# In[19]:


# Python program to generate WordCloud 

# Text of all words in column bloom

text = " ".join(word for word in data['text'])
print ("There are {} words in the combination of all cells in all books.".format(len(text)))

# Generate a word cloud image

wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)

# Display the generated image:
# the matplotlib way:

plt.axis("off")
plt.figure( figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()


# ### Histogram of text length of each writer

# In[20]:


def length(text):    
    '''a function which returns the length of text'''
    return len(text)


# In[21]:


data['length'] = data['text'].apply(length)
data.head(10)


# In[22]:


EAP_data = data[data['author'] == 'EAP']
SACD_data = data[data['author'] == 'SACD']
AC_data = data[data['author'] == 'AC']


# In[23]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
bins = 500
plt.hist(EAP_data['length'], alpha = 0.6, bins=bins, label='Edgar Allan Poe')
plt.hist(SACD_data['length'], alpha = 0.8, bins=bins, label='Sir Arthur Conan Doyle')
plt.hist(AC_data['length'], alpha = 0.4, bins=bins, label='Agatha Christie')
plt.xlabel('length')
plt.ylabel('numbers')
plt.legend(loc='upper right')
plt.xlim(0,300)
plt.grid()
plt.show()


# ## Top Words Of Each Writer

# ### Edgar Allan Poe

# In[24]:


vec = CountVectorizer().fit(EAP_data['text'])
bag_of_words = vec.transform(EAP_data['text'])
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
words_freq[:10]

from pandas import DataFrame
Top_10_words = DataFrame (words_freq[:10],columns=['word', 'total'])
Top_10_words


# In[25]:


import numpy as np                                                               
import matplotlib.pyplot as plt

#top=[('a',1.875),('c',1.125),('d',0.5)]
words_freq[:10]

labels, ys = zip(*words_freq[:10])
xs = np.arange(len(labels)) 
width = 1

plt.figure(figsize=(10,5))
plt.bar(xs, ys, width=0.7, align='center', color='red')

plt.xticks(xs, labels) #Replace default x-ticks with xs, then replace xs with labels
plt.yticks(ys)


# In[26]:


# Python program to generate WordCloud 

# Text of all words in column bloom

text = " ".join(word for word in EAP_data['text'])
print ("There are {} words in the combination of all cells in all books.".format(len(text)))

# Generate a word cloud image

wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)

# Display the generated image:
# the matplotlib way:

plt.axis("off")
plt.figure( figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()


# ### Sir Arthur Conan Doyle

# In[27]:


vec = CountVectorizer().fit(SACD_data['text'])
bag_of_words = vec.transform(SACD_data['text'])
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
words_freq[:10]

from pandas import DataFrame
Top_10_words = DataFrame (words_freq[:10],columns=['word', 'total'])
Top_10_words


# In[28]:


import numpy as np                                                               
import matplotlib.pyplot as plt

#top=[('a',1.875),('c',1.125),('d',0.5)]
words_freq[:10]

labels, ys = zip(*words_freq[:10])
xs = np.arange(len(labels)) 
width = 1

plt.figure(figsize=(10,5))
plt.bar(xs, ys, width=0.7, align='center', color='green')

plt.xticks(xs, labels) #Replace default x-ticks with xs, then replace xs with labels
plt.yticks(ys)


# In[29]:


# Python program to generate WordCloud 

# Text of all words in column bloom

text = " ".join(word for word in SACD_data['text'])
print ("There are {} words in the combination of all cells in all books.".format(len(text)))

# Generate a word cloud image

wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)

# Display the generated image:
# the matplotlib way:

plt.axis("off")
plt.figure( figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()


# ### Agatha Christie

# In[30]:


vec = CountVectorizer().fit(AC_data['text'])
bag_of_words = vec.transform(AC_data['text'])
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
words_freq[:10]

from pandas import DataFrame
Top_10_words = DataFrame (words_freq[:10],columns=['word', 'total'])
Top_10_words


# In[31]:


import numpy as np                                                               
import matplotlib.pyplot as plt

#top=[('a',1.875),('c',1.125),('d',0.5)]
words_freq[:10]

labels, ys = zip(*words_freq[:10])
xs = np.arange(len(labels)) 
width = 1

plt.figure(figsize=(10,5))
plt.bar(xs, ys, width=0.7, align='center', color='blue')

plt.xticks(xs, labels) #Replace default x-ticks with xs, then replace xs with labels
plt.yticks(ys)


# In[32]:


# Python program to generate WordCloud 

# Text of all words in column bloom

text = " ".join(word for word in AC_data['text'])
print ("There are {} words in the combination of all cells in all books.".format(len(text)))

# Generate a word cloud image

wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)

# Display the generated image:
# the matplotlib way:

plt.axis("off")
plt.figure( figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()


# ## TF-IDF Extraction

# In[33]:


# create the object of tfid vectorizer
tfid_vectorizer = TfidfVectorizer("english")
# fit the vectorizer using the text data
tfid_vectorizer.fit(data['text'])
# collect the vocabulary items used in the vectorizer
dictionary = tfid_vectorizer.vocabulary_.items()  
# extract the tfid representation matrix of the text data
tfid_matrix = tfid_vectorizer.transform(data['text'])
# collect the tfid matrix in numpy array
array = tfid_matrix.todense()


# In[53]:


# store the tf-idf array into pandas dataframe
df = pd.DataFrame(array)
df


# ### Training Model

# In[35]:


df['output'] = data['author']
df['id'] = data['id']
df.head(10)


# In[36]:


features = df.columns.tolist()
output = 'output'
# removing the output and the id from features
features.remove(output)
features.remove('id')


# In[37]:


from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV


# ### Tuning Multinomial Naive Bayes Classifier

# In[38]:


alpha_list1 = np.linspace(0.006, 0.1, 20)
alpha_list1 = np.around(alpha_list1, decimals=4)
alpha_list1


# In[39]:


# parameter grid
parameter_grid = [{"alpha":alpha_list1}]


# In[40]:


# classifier object
classifier1 = MultinomialNB()
# gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter
gridsearch1 = GridSearchCV(classifier1,parameter_grid, scoring = 'neg_log_loss', cv = 4)
# fit the gridsearch
gridsearch1.fit(df[features], df[output])


# In[41]:


results1 = pd.DataFrame()
# collect alpha list
results1['alpha'] = gridsearch1.cv_results_['param_alpha'].data
# collect test scores
results1['neglogloss'] = gridsearch1.cv_results_['mean_test_score'].data


# In[42]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
plt.plot(results1['alpha'], -results1['neglogloss'])
plt.xlabel('alpha')
plt.ylabel('logloss')
plt.grid()


# In[43]:


print("Best parameter: ",gridsearch1.best_params_)


# In[44]:


print("Best score: ",gridsearch1.best_score_) 


# ### Tuning Multinomial Naive Bayes Classifier

# In[45]:


alpha_list2 = np.linspace(0.006, 0.1, 20)
alpha_list2 = np.around(alpha_list2, decimals=4)
alpha_list2


# In[46]:


parameter_grid = [{"alpha":alpha_list2}]


# In[47]:


# classifier object
classifier2 = MultinomialNB()
# gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter
gridsearch2 = GridSearchCV(classifier2,parameter_grid, scoring = 'neg_log_loss', cv = 4)
# fit the gridsearch
gridsearch2.fit(df[features], df[output])


# In[48]:


# classifier object
classifier2 = MultinomialNB()
# gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter
gridsearch2 = GridSearchCV(classifier2,parameter_grid, scoring = 'neg_log_loss', cv = 4)
# fit the gridsearch
gridsearch2.fit(df[features], df[output])


# In[49]:


results2 = pd.DataFrame()
# collect alpha list
results2['alpha'] = gridsearch2.cv_results_['param_alpha'].data
# collect test scores
results2['neglogloss'] = gridsearch2.cv_results_['mean_test_score'].data


# In[50]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
plt.plot(results2['alpha'], -results2['neglogloss'])
plt.xlabel('alpha')
plt.ylabel('logloss')
plt.grid()


# In[51]:


print("Best parameter: ",gridsearch2.best_params_)


# In[52]:


print("Best score: ",gridsearch2.best_score_)

