# -*- coding: utf-8 -*-
## @brief Created on Tue May  5 00:25:31 2020
#
#@author: Priya
#


#	heuristic to check if commit is from different project
#	Build a binary classifier to classify commits from different projects

#	import all libraies required to build binary and multiclass classifiers
#	import libray for python data structures, array, dataframe

import pandas as pd
import numpy as np

#	import libararies from NLTK package for text pre-processing


from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#	import libary to encode classes and refactoring type

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn

#	import libraries to perform feature extraction , TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer

#	import libraries hving inbuilt fnctions of classifiers

from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('C:/Users/Priya/Desktop/Capstone_Commits_sem2/test_projectclass.csv')
df

#	check NAN in datafarme
#	class column has 25 NAN entries, remove them

df.isna().sum()
df.dropna()


#	drop nan values from commit message column

df['Message'].dropna(inplace=True)


def lower_text(text):#    normalize text for further processing#    function to normalize text for further processing# 	input	text(commit messages from datafarme)# 	output	text in lower cases##
    text = [entry.lower() for entry in text]
    return text

df['Message']= df['Message'].apply(lambda x: lower_text(x))

#	Perform tokenization on commit messages
df['Message']= [word_tokenize(entry) for entry in df['Message']]


#	define varibles for adjetive, verb and adverb

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(df['Message']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'commit_text'
    df.loc[index,'commit_text'] = str(Final_words)


#	Heuristic 1: How to check if commit is not from the same project?
#	Build a binary classifier which will classify commits from another proejct
#	Here I am using Project as my class
#	Split training and testing data set

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['commit_text'],df['Project'],test_size=0.3)

#Encode project class using encoder library
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

#	convert train and test commit message text data into TFIDF vectors
#	TFIDF: combine word embeddings with TFIDF values

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['commit_text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)



#	Print TFIDF Vector, vocabualry used to calculate tfidf values

print(Tfidf_vect.vocabulary_)

 #	Print vectorized data
 
print(Train_X_Tfidf)


#	Test performance with different classifiers, Naive bayes, logistic regression, Random Forest, SVM
#	Build a Naive bayes classifier

#fit the training dataset on the NB classifier
#
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)

#	Prefict project class for test data
predictions_NB = Naive.predict(Test_X_Tfidf)

#	Print accuracy of naive bayes classifier
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


#	Analyze results:
#	confusin matrix for binary naive bayes classifier
#	code chanegs with commit text
#	import libary to calculate confusion matrix

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Test_Y, predictions_NB)
cnf_matrix


#	Plot confusion matrix for naive bayes classifier uisng matplotlib#	import required modules
 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix of Naive Bayes classifier', y=1.1)
plt.ylabel('Actual Project label')
plt.xlabel('Predicted Project label')

#Print accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(Test_Y, predictions_NB))
print("Precision:",metrics.precision_score(Test_Y, predictions_NB))
print("Recall:",metrics.recall_score(Test_Y, predictions_NB))

#
#	Build a SVM classifier
#	fit the training dataset on the SVM classifier
#	Classifier - Algorithm - SVM
#
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

#	Prefict project class for test data
pred_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(pred_SVM, Test_Y)*100)


#	confusin matrix for binary SVM classifier
#	code chanegs with commit text
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Test_Y, pred_SVM)
cnf_matrix


#	Plot confusion matrix for SVM uisng matplotlib
#	import required modules
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for SVM', y=1.1)
plt.ylabel('Actual Project label')
plt.xlabel('Predicted Project label')

#import libraries required to print accuracy score and result matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(Test_Y,pred_SVM))
print(classification_report(Test_Y,pred_SVM))
print(accuracy_score(Test_Y, pred_SVM))


#	Build a classifier using Random Forest Approach
#	fit the training dataset on the RF classifier
#	import libraries having built in classifier functions
#
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(Train_X_Tfidf,Train_Y)

#Prefict project class for test data using RF classifier
rf_predict = rfc.predict(Test_X_Tfidf)

#
#	Analyze the model and perform cross validation
#	import libraries for calculate cross_val_score,generate classification report
#
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
print("Confusion Matrix for RF classifier/ classify project class")
print(confusion_matrix(Test_Y, rf_predict))
print('\n')

#Print classification report
print(classification_report(Test_Y, rf_predict))
print('\n')

#
#	Plot confusion matrix for RF uisng matplotlib
#	import required modules
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for RF classifier', y=1.1)
plt.ylabel('Actual Project label')
plt.xlabel('Predicted Project label')


#Build a classifier using logistic regression
# fit the training dataset on the classifier
#import libraries having built in classifier functions
#logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(Train_X_Tfidf,Train_Y)

#Prefict project class for test data
regres_pred=logreg.predict(Test_X_Tfidf)

print("Confusion Matrix for Logistic Regression/ classify project class")
cnf_matrix = metrics.confusion_matrix(Test_Y, regres_pred)
cnf_matrix

#Plot heat map of confusion matrix/ Logistic regression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Logistic Regession', y=1.1)
plt.ylabel('Actual Project label')
plt.xlabel('Predicted Project label')

#
#	plot ROC curve for data
#	ROC Curve : Receiver opertaor characteristics
#	plot of true postive rate againt false postive rate
#	plot to analyze binary classifier
#
y_pred_proba = logreg.predict_proba(Test_X_Tfidf)[::,1]
fpr, tpr, _ = metrics.roc_curve(Test_Y,  y_pred_proba)
auc = metrics.roc_auc_score(Test_Y, y_pred_proba)
plt.plot(fpr,tpr,label="ROC Curve for data, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# -*- coding: utf-8 -*-
#
#	Created on Tue May  5 02:30:57 2020
#
#	@author: Priya
#

#
#	heuristic to check refactoring type of commit ( why developer commited any change in repository)
#	Build a binary classifier to classify commits based of refactoring type
#	Here I am using Refactoring type as my class
#	Split training and testing data set
#
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['commit_text'],df['refactoring_type'],test_size=0.3)


#Encode refactoring type using encoder library
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


#	convert train and test commit message text data into TFIDF vectors
#	TFIDF: combine word embeddings with TFIDF values
#
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['commit_text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


#	Test performance with different classifiers, Naive bayes, logistic regression, Random Forest, SVM
#	Build a Naive bayes classifier
#	fit the training dataset on the NB classifier/ Refactoring type#

Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)

#Prefict refactoring type for test data
predictions_NB = Naive.predict(Test_X_Tfidf)

#Print accuracy of naive bayes classifier
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


#	steps:
#	Analysis of results
#	confusin matrix for binary naive bayes classifier
#	code chanegs with commit text
#	import libary to calculate confusion matrix#

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Test_Y, predictions_NB)
cnf_matrix


#	Plot confusion matrix for naive bayes classifier uisng matplotlib
#	import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix of Naive Bayes classifier', y=1.1)
plt.ylabel('Actual Refactoring type label')
plt.xlabel('Predicted Refactoring type  label')


#Build a SVM classifier
#fit the training dataset on the SVM classifier
#Classifier - Algorithm - SVM


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

#Prefict project class for test data
pred_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(pred_SVM, Test_Y)*100)


#Plot confusion matrix for SVM uisng matplotlib
#import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for SVM', y=1.1)
plt.ylabel('Actual Refactoring type label')
plt.xlabel('Predicted Refactoring type label')


#import libraries required to print accuracy score and result matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(Test_Y,pred_SVM))
print(classification_report(Test_Y,pred_SVM))
print(accuracy_score(Test_Y, pred_SVM))

#Build a classifier using Random Forest Approach
# fit the training dataset on the RF classifier
#import libraries having built in classifier functions
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(Train_X_Tfidf,Train_Y)

#Prefict refactoring type for test data using RF classifier
rf_predict = rfc.predict(Test_X_Tfidf)

#Analyze the model and perform cross validation
#import libraries for calculate cross_val_score,generate classification report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
print("Confusion Matrix for RF classifier/ classify Refactoring type class")
print(confusion_matrix(Test_Y, rf_predict))
print('\n')

#Print classification report
print(classification_report(Test_Y, rf_predict))
print('\n')


#	Plot confusion matrix for RF uisng matplotlib
#	import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for RF classifier', y=1.1)
plt.ylabel('Actual Refactoring type label')
plt.xlabel('Predicted Refactoring type label')


#	Build a classifier using logistic regression
#	fit the training dataset on the classifier
#	import libraries having built in classifier functions
#	logistic regression
#
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(Train_X_Tfidf,Train_Y)

#Prefict refactoring type for test data
regres_pred=logreg.predict(Test_X_Tfidf)

print("Confusion Matrix for Logistic Regression/ classify Refactoring type")
cnf_matrix = metrics.confusion_matrix(Test_Y, regres_pred)
cnf_matrix


#Plot heat map of confusion matrix/ Logistic regression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Logistic Regession', y=1.1)
plt.ylabel('Actual Refactoring type label')
plt.xlabel('Predicted Refactoring type label')


# Analyze data :  commit time
# Heuristic 3: to check commits sent at unusual times

#convert authordate_text into datetime type
df['AuthorDate_Text'] = pd.to_datetime(df['AuthorDate_Text'])

#split time
df['new_time'] = [d.time() for d in df['AuthorDate_Text']]

#split date
df['new_date'] = [d.date() for d in df['AuthorDate_Text']]


#plot line graph to analyze pattern / commit time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
plt.plot(df['Id'],df['new_time'])
plt.gcf().autofmt_xdate()
plt.title('Commit ids vs time')
plt.ylabel('commit time')
plt.xlabel('Commit Ids')
plt.show()

#plot line graph to analyze pattern / commit date
plt.plot(df['Id'],df['new_date'])
plt.gcf().autofmt_xdate()
plt.title('Commit ids vs date of commit')
plt.ylabel('commit date')
plt.xlabel('Commit Ids')
plt.show()



#	plot boxplot of time to detect outliers in commit time
columns = ['new_time', 'new_date']
df1 = df.reindex(columns=columns)

#import libraries required to plot boxplot from seaborn package
import seaborn as sns
# import matplotlib library
import matplotlib.pyplot as plt

a4_dims = (15.7, 10.27)
fig, ax = pyplot.subplots(figsize=a4_dims)

sns.boxplot(ax=ax,x=df.new_time, y=df.Id)


#plot boxplot to detect outliers for date of commits

a4_dims = (15.7, 10.27)
fig, ax = pyplot.subplots(figsize=a4_dims)

sns.boxplot(ax=ax,x=df.new_data, y=df.Id)


#	Heuristic 4: check for commit from outstanding commiter
#	detect commits from commiter who have contributed less than 2 times in project
#	can we consider commits as unusual if commiter has commited only once??

df['AuthorEmail']
count_auth= df.AuthorEmail.str.split(expand=True).stack().value_counts() <2
count_auth


#	detect unusual commits


#df.loc[df['AuthorEmail'].isin('dan@metabroadcase.com')]
df.loc[df['AuthorEmail'] == 'dan@metabroadcast.com']
