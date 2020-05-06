# CommitMessageClassification
 This Project aims to support developers and managers in understanding the maintenance activities performed in their code repositories by analyzing  the source code repository to identify unusual commits and classify those into  different maintenance activities, already added by developers in their version control systems
 
## Data Set used in this project:

For this project, commits from projects written in java, C, C++ is used. To fetch data from GitProc we need to provide   GitHub project URL and this software gives .sql files of data. Data includes commits, author id, date, time of commit, refactoring details, classes, ﬁle path, etc.

* Data directory has test data file containing commits for projects written in C, C++, Java. It is publicly accessible. The actual file I am using for this project is not publicly accessible.


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


## How to install required text- processing libraries? 

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
## How to install SQLite DB? 
SQLite DB is required to access the dataset.

use the following link to download this software:
(https://sqlitebrowser.org/dl/)

## Once all the softwares are successfully installed you can run these scripts from respository:
## Steps to run these scripts:
* Using jupyter
1. Open Jupyter
2.  Open these files 
3. Run 

* Using Command line
1. use command 
```
python filename.py

```

* Using IDE (PyCharm / Spyder)
1. Open these files 
2. Run as per instructions ( https://www.jetbrains.com/help/pycharm/creating-and-running-your-first-python-project.html)

## Documentation
* Documentation of this project is in the following directory :
(https://github.com/priyadarshni/CommitMessageClassification/blob/master/doc/_build/html/index.html) 
