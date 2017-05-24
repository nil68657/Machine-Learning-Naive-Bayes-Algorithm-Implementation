# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:32:18 2017

@author: nilan
"""
import pandas as pd
import re
import csv
import string
from collections import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score as cvs
import matplotlib.pyplot as plt



def extract_optical_xls_file(file_name):
        try:
            file = open(file_name, 'r')
            data_set = pd.read_csv(file)
            features = list(data_set.columns[:-1])
            data = data_set[features]
            target = data_set.columns[-1]
            result = data_set[target]
            file.close()
            return {
                'data': data,
                'result': result
            }
        except IOError as error:
            print('IOError: ' + error.args[1])


def optical_nbay(training_file, test_file):
    training = extract_optical_xls_file(training_file)
    test = extract_optical_xls_file(test_file)
    clf = GaussianNB()
    clf.fit(training['data'], training['result'])
    print(clf.score(test['data'], test['result']))
    print(cvs(clf, X=test['data'], y=test['result'], verbose=1,cv=5))

#C:\Users\NIlan\Desktop\.csv
#C:\Users\nilan\Desktop\ML/optdigits_raining.csv
optical_nbay('C:\\Users\\nilan\\Desktop\\ML\\optdigits_raining.csv', 'C:\\Users\\nilan\\Desktop\\ML\\optdigits_test.csv')