.. CommitClassifier documentation master file, created by
   sphinx-quickstart on Wed May  6 02:59:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CommitClassifier's documentation!
============================================
Goal of this Project:
---------------------
This Project aims to support developers and managers in understanding the maintenance activities performed in their code repositories by analyzing the
source code repository to identify unusual commits and classify those into different maintenance activities, already added by developers in their version control systems.

Different Modules from the project:
-----------------------------------

Data Ingestion :   Data for the proposed project is extracted using GiProc software. For this project,
commits from projects written in java, C, C++ is used. To fetch data from Git Proc we need to provide GitHub project URL and this software gives ﬂat ﬁles 
of data. Data includes commits, author id, date, time of commit, refactoring details, classes, ﬁle path, etc. The ﬁnal model will be developed for 30+ projects 
written in C, C++, and java. For the temporary purpose, data is being stored in the SQL server. Data stored in the SQL server is raw data and it’s notusableformodeldevelopment,
thus data cleaning is required. Every commit has attributes like commitid,message,authorname,id,ﬁlepath.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Data Cleaning: Text-Preprocessing: The goal of the data cleaning layer is to prepare data for the modeling layer
--------------------------------------------------------------------------------------------------------------------------------------------------

Data Analysis  : Analyze data to find out interesting insights from data, in commitclasssifier.py file, you can find I have analyzed commits using date , 
time , commiter email , location
------------------------------------------------------------------------------------------------------------------------------------------------------------


Data Analysis  : Analyze data to find out interesting insights from data, in commitclasssifier.py file, you can find I have analyzed commits using date , 
time , commiter email , location
--------------------------------------------------------------------------------------------------------------------------------------------------------------


Data Modelling
-----------------------------------


Validation
-----------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:
    Commit-Text Pre-Processing
	Commit Message Classifier


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
