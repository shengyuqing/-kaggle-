# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:16:51 2016

@author: Administrator
"""
import graphlab
sales = graphlab.SFrame('kc_house_data.gl/')

from math import log, sqrt

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to float, before creating a new feature.
sales['floors'] = sales['floors'].astype(float) 
sales['floors_square'] = sales['floors']*sales['floors']

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = graphlab.linear_regression.create(sales, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty=1e10)

model_all.get('coefficients').print_rows(num_rows=18, num_columns=4)

(training_and_validation, testing) = sales.random_split(.9,seed=1) # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1) # split training into train and validate

import numpy as np
l1_penalty = np.logspace(1, 7, num=13) 

rss_list = []


for num in l1_penalty:
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty= num,verbose = False)
    print model.get('coefficients')
    rss = sum((validation['price'] - model.predict(validation))**2)
    rss_list.append(rss)

from matplotlib import pyplot as plt


model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty= 10,verbose = False)
print model.get('coefficients').print_rows(num_rows=18, num_columns=4)

max_nonzeros = 7
l1_penalty_values = np.logspace(8, 10, num=20)

rss_list1 = []
nozero_list = []
for num in l1_penalty_values:
    model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty= num,verbose = False)
    non_zeros = model['coefficients']['value'].nnz()
    nozero_list.append(non_zeros)
    
    rss = sum((validation['price'] - model.predict(validation))**2)
    print rss
    rss_list.append(rss)
    
        
    '''
    rss = sum((validation['price'] - model.predict(validation))**2)
    rss_list1.append(rss)
    print rss_list1
    '''
l1_penalty_min = 
l1_penalty_max =   
  
l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)