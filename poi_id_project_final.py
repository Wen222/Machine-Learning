
# coding: utf-8


#!/usr/bin/python

import os
import sys
sys.path.append("../../tools/")
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import tester
import numpy as np
import pandas as pd
from time import time


# Load data and store as pandas dataframe

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Remove outlier
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


df = pd.DataFrame(data_dict).T
print df.shape

# Change data types
df['bonus'] = df['bonus'].astype('float64')
df['deferral_payments'] = df['deferral_payments'].astype('float64')
df['deferred_income'] = df['deferred_income'].astype('float64')
df['director_fees'] = df['director_fees'].astype('float64')
df['email_address'] = df['bonus'].astype('str')
df['exercised_stock_options'] = df['exercised_stock_options'].astype('float64')
df['expenses'] = df['expenses'].astype('float64')
df['from_messages'] = df['from_messages'].astype('float64')
df['from_poi_to_this_person'] = df['from_poi_to_this_person'].astype('float64')
df['from_this_person_to_poi'] = df['from_this_person_to_poi'].astype('float64')
df['loan_advances'] = df['loan_advances'].astype('float64')
df['long_term_incentive'] = df['long_term_incentive'].astype('float64')
df['other'] = df['other'].astype('float64')
df['poi'] = df['poi'].astype('bool')
df['restricted_stock'] = df['restricted_stock'].astype('float64')
df['restricted_stock_deferred'] = df['restricted_stock_deferred'].astype('float64')
df['salary'] = df['salary'].astype('float64')
df['shared_receipt_with_poi'] = df['shared_receipt_with_poi'].astype('float64')
df['to_messages'] = df['to_messages'].astype('float64')
df['total_payments'] = df['total_payments'].astype('float64')
df['total_stock_value'] = df['total_stock_value'].astype('float64')



print df.dtypes

df.describe()



payment_fields = ['salary',
                  'bonus', 
                  'long_term_incentive', 
                  'deferred_income',
                  'deferral_payments',
                  'loan_advances',
                  'other',
                  'expenses',
                  'director_fees',
                  'total_payments']

stock_fields = ['exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value']

email_data = ['from_messages',
              'from_poi_to_this_person',
              'from_this_person_to_poi',
              'shared_receipt_with_poi',
              'to_messages']
df[payment_fields] = df[payment_fields].fillna(0)
df[stock_fields] = df[stock_fields].fillna(0)


imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
df_poi = df[df['poi'] == True]
df_nonpoi = df[df['poi'] == False]

df_poi.ix[:,email_data] = imp.fit_transform(df_poi.ix[:,email_data]);
df_nonpoi.ix[:,email_data] = imp.fit_transform(df_nonpoi.ix[:,email_data]);

df = df_poi.append(df_nonpoi)



# First include all relevent features
features_list = ['poi',
                 'bonus',
                 'salary',
                 'deferral_payments',
                 'deferred_income',
                 'director_fees',
                 'exercised_stock_options',
                 'expenses',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'loan_advances',
                 'long_term_incentive',
                 'other', 
                 'restricted_stock',
                 'restricted_stock_deferred', 
                 'shared_receipt_with_poi',
                 'to_messages', 
                 'total_payments', 
                 'total_stock_value']



# Seperate labels and features
data_dict_cleaned = df.to_dict(orient = 'index')
my_dataset = df.to_dict(orient = 'index')
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)



# Create new features
for v in data_dict_cleaned.values():
    from_poi_to_this_person = v["from_poi_to_this_person"]
    to_messages = v["to_messages"]
    from_this_person_to_poi = v["from_this_person_to_poi"]
    from_messages = v["from_messages"]
    shared_receipt_with_poi = v['shared_receipt_with_poi']
    bonus = v['bonus']
    salary = v['salary']
    total_payments = v['total_payments']

    v["from_poi_ratio"] = (float(from_poi_to_this_person) / float(to_messages) if
                           to_messages not in [0, "NaN"] and from_poi_to_this_person
                           not in [0, "NaN"] else 0.0)
    v["to_poi_ratio"] = (float(from_this_person_to_poi) / float(from_messages) if
                         from_messages not in [0, "NaN"] and from_this_person_to_poi
                         not in [0, "NaN"] else 0.0)
    v["shared_poi_ratio"] = (float(shared_receipt_with_poi) / float(to_messages) if
                         to_messages not in [0, "NaN"] and shared_receipt_with_poi
                         not in [0, "NaN"] else 0.0)
    v["bonus_salary_ratio"] = (float(bonus) / float(salary) if
                         salary not in [0, "NaN"] and bonus
                         not in [0, "NaN"] else 0.0)
    v["bonus_ratio"] = (float(bonus) / float(total_payments) if
                         total_payments not in [0, "NaN"] and bonus
                         not in [0, "NaN"] else 0.0)

features_list.append("from_poi_ratio")
features_list.append("to_poi_ratio")
features_list.append("shared_poi_ratio")
features_list.append("bonus_salary_ratio")
features_list.append("bonus_ratio")
#print features_list
features_list_all = features_list
print features_list_all



# Re-orgnize dataset
my_dataset = data_dict_cleaned
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)


# Feature Selection
features_list = []
features_list.append("poi")
features_list.append("expenses")
features_list.append("other")
features_list.append("exercised_stock_options")
features_list.append("to_poi_ratio")
features_list.append("shared_receipt_with_poi")

    
print features_list

clf = DecisionTreeClassifier()
tester.dump_classifier_and_data(clf,my_dataset,features_list)
tester.main();


