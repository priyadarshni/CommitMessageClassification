## @package nltk
#  loading libraries
#
#loading pandas
import pandas as pd

#loading numpy
import numpy as np

#loading NLTK text precessing libaray
import nltk

#import list for list of punctuations
import string

#Import beautifulSoup

from bs4 import BeautifulSoup

#import stop word list

from nltk.corpus import stopwords

#import tokenizer 

from nltk.tokenize import RegexpTokenizer

#import Lemmatizer

from nltk.stem import WordNetLemmatizer

#import stemmer

from nltk.stem.porter import PorterStemmer



#creating dataframe of commits

df = pd.read_csv('C:/Users/Priya/Downloads/miner_refactoring.csv')


#Remove duplicate commit_ids as a part of EDA
df = (df.drop_duplicates(['CommitId'], keep ='last'));

#display dataframe
df



#drop NA values from commit text
commit_text= df['Message'].dropna()



#remove duplicate commitId's
df = (df.drop_duplicates(['CommitId'], keep ='last'));



#        This function is for removing html tags from commits
# 	input	commit messages in data frame
# 	output	commit messages without html tags in dataframe
#

def remove_html(text):

    soup = BeautifulSoup(text, 'lxml')

    hrml_free = soup.get_text()

    return hrml_free




#        function to remove punctuations
#        This function will removes punctuations from commit messages
# 	input	commit messages in data frame
# 	output	commit messages without punctuations in dataframe
#

def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])

    return no_punct



#import string

dir(string)

#give call to remove_punctations()
commit_text= commit_text.apply(lambda x: remove_punctuation(x))

commit_text.head()

#dispaly commit_text dataframe
commit_text


#tokenize

#instantiate tokenizer

#split up by spaces

tokenizer = RegexpTokenizer(r'\w+')

#call to tokenizer
commit_text = commit_text.apply(lambda x: tokenizer.tokenize(x.lower()))

#display top 100 commit messages
commit_text.head(100)



#        remove stop words
#        this function will remove stopwords from commit messages
#        input:commit messages in data frame
#        output:text without stopwords
#

def remove_stopwords(text):

    words = [w for w in text if w not in stopwords.words('english')]

    return words



#remove stop words from english
commit_text = commit_text.apply(lambda x : remove_stopwords(x))



#print starting intial commits
commit_text.head(100)



#Lemmatization

lemmatizer = WordNetLemmatizer()

#        this function will is for Lemmatizing,
#        on the other hand, maps common words into one base.
# 	input	commit messages in data frame
# 	output	commit message with shorten words back to their root form
#

def word_lemmatizer(text):
    for_text= [lemmatizer.lemmatize(i) for i in text]

    return for_text


#call to lammetization
commit_text.apply(lambda x :word_lemmatizer(x))



#stemming
stemmer= PorterStemmer()


#         this function will perform stemming on commit messages
# 	input	commit messages in data frame
#         output:text with stemmed words
#

def word_stemmer(text):
    stem_commits = "".join([stemmer.stem(i) for i in text])

    return stem_commits



commit_text = commit_text.apply(lambda x : word_stemmer(x))

#count frequency of words

commit_text.str.split(expand=True).stack().value_counts()




#copy commit messages to text file
df.to_csv(r'C:\Users\Priya\Desktop\Capstone_Commits_sem2\commits.txt', header=None, index=None, sep=' ', mode='a')



from collections import Counter



#opens the file. the with statement here will automatically close it afterwards.

with open('C:\\Users\\Priya\\Desktop\\Capstone_Commits_sem2\\commits.txt',encoding="utf8") as input_file:

    #build a counter from each word in the file

    count = Counter(word for line in input_file

                         for word in line.split())



print(count.most_common(200))





#count most frequent words

frequent_words = count.most_common(200)



#create  a dataframe of frequent words

freq_w = pd.DataFrame(frequent_words)



words = freq_w[0]

#created a dictioanry of frequent words

data_dict = words.to_dict()
