# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:44:28 2016

@author: Administrator
"""
from __future__ import division
import graphlab
import numpy as np
graphlab.canvas.set_target('ipynb')

products = graphlab.SFrame('amazon_baby.gl/')


#删除标点符号函数
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

#删除标点符号
review_clean = products['review'].apply(remove_punctuation)

# 计算词频并生成新特征
products['word_count'] = graphlab.text_analytics.count_words(review_clean)

# 删除打分为3的评分
products = products[products['rating'] != 3]

# 大于3的做为正特征，小于3作为负特征
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
#划分数据集
train_data, test_data = products.random_split(.8, seed=1)
#建立模型
model = graphlab.logistic_classifier.create(train_data, target='sentiment',
                                            features=['word_count'],
                                            validation_set=None)
                                            
accuracy= model.evaluate(test_data, metric='accuracy')['accuracy']
print "测试集上模型的准确率为: %s" % accuracy

#测试集上正面情绪用户所占比例
baseline = len(test_data[test_data['sentiment'] == 1])/float(len(test_data))
print "测试集上正面情绪用户所占比例: %s" % baseline

#打印出精准率和召回率
confusion_matrix = model.evaluate(test_data, metric='confusion_matrix')['confusion_matrix']
confusion_matrix

precision = model.evaluate(test_data, metric='precision')['precision']
print "测试集上精准率为: %s" % precision

recall = model.evaluate(test_data, metric='recall')['recall']
print "测试集上召回率为: %s" % recall

#阈值函数
def apply_threshold(probabilities, threshold):
    predictions = probabilities >= threshold
    return predictions.apply(lambda x: 1 if x == 1 else -1)
    
    
probabilities = model.predict(test_data, output_type='probability')
predictions_with_default_threshold = apply_threshold(probabilities, threshold = 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)

print "threshold = 0.5时: %s" % (predictions_with_default_threshold == 1).sum()
print "threshold = 0.9时: %s" % (predictions_with_high_threshold == 1).sum()

# Threshold = 0.5时
precision_with_default_threshold = graphlab.evaluation.precision(test_data['sentiment'],
                                        predictions_with_default_threshold)

recall_with_default_threshold = graphlab.evaluation.recall(test_data['sentiment'],
                                        predictions_with_default_threshold)

# Threshold = 0.9时
precision_with_high_threshold = graphlab.evaluation.precision(test_data['sentiment'],
                                        predictions_with_high_threshold)
recall_with_high_threshold = graphlab.evaluation.recall(test_data['sentiment'],
                                        predictions_with_high_threshold)
       


                                 
threshold_values = np.linspace(0.5, 1, num=100)
print threshold_values 


precision_all = []
recall_all = []

probabilities = model.predict(test_data, output_type='probability')
for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    
    precision = graphlab.evaluation.precision(test_data['sentiment'], predictions)
    recall = graphlab.evaluation.recall(test_data['sentiment'], predictions)
    
    precision_all.append(precision)
    recall_all.append(recall)
    
import matplotlib.pyplot as plt
%matplotlib inline

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})
    
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')

for t, p in zip(threshold_values, precision_all):
    if p >= 0.965:
        print t,p
        break

predictions_098 = apply_threshold(probabilities, threshold = 0.98)
graphlab.evaluation.confusion_matrix(test_data['sentiment'], predictions_098)








baby_reviews =  test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]
probabilities = model.predict(baby_reviews, output_type='probability')
threshold_values = np.linspace(0.5, 1, num=100)

precision_all = []
recall_all = []

for threshold in threshold_values:
    
    predictions = apply_threshold(probabilities, threshold)

    precision = graphlab.evaluation.precision(baby_reviews['sentiment'], predictions)    

    recall = graphlab.evaluation.recall(baby_reviews['sentiment'], predictions)
    precision_all.append(precision)
    recall_all.append(recall)
    
for t, p in zip(threshold_values, precision_all):
    if p >= 0.965:
        print t,p
        break
    
