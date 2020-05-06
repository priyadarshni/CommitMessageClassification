# CommitMessageClassification
 this Project aims to support developers and managers in understanding the maintenance activities performed in their code repositories by analyzing  the source code repository to identify unusual commits and classify those into  different maintenance activities, already added by developers in their version control systems

## How to run the files from CommitMessageClassification repository to generate results?

## Software Requirements
* Anaconda : Python 
* Text processing libraries 
* SQLite DB 

## How to install Anaconda?
* On Windows machine:
Follow the steps :

Install Anaconda on Windows 10(https://docs.anaconda.com/anaconda/install/windows/)

* On Mac machine :

Install Anaconda on Mac (https://docs.anaconda.com/anaconda/install/mac-os/)

**If you follow the similar steps given on above websites you will have different python editors in your system installed**


# How to install required text- processing libraries? 

* Run Python interpreter on Windows/ Linux / Mac
1. Enter the command 
  ```
  import nltk
  nltk.download ()
  ```
 2. Follow steps( https://www.guru99.com/download-install-nltk.html)
 3. Test if NLTK is installed properly on your system 
 ```
 from nltk.corpus import brown
 brown.words()
 
 ```
# How to install SQLite DB? 
SQLite DB is required to access the dataset.

use the following link to download this software:
(https://sqlitebrowser.org/dl/)
