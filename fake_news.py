#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np
import pandas as pd


# In[92]:


#read train csv and split
delimiter="\t"
df = pd.read_csv('train.csv',delimiter="\t").astype(str)

df.head(10)


# In[93]:


#find str in label
fliter = (df['label']== 'label')
df[fliter]


# In[94]:


df= df.drop(index=1615)


# In[95]:


data_x= df['text']
data_y= df['label']


# In[96]:


#read test csv and split
delimiter="\t"
df2 = pd.read_csv('test.csv',delimiter="\t").astype(str)

df2.head(10)


# In[97]:


test_x= df2['text']


# In[98]:


df3=pd.read_csv('sample_submission.csv')
df3.head()


# In[99]:


test_y=df3['label']


# In[100]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[101]:


#tfidf weights
tfidf = TfidfVectorizer(stop_words='english')


# In[102]:


#the text in train&test data turn into tfidf weight
data_x_tfidf = tfidf.fit_transform(data_x)
test_x_tfidf = tfidf.fit_transform(test_x)


# In[103]:


data_x_tfidf 


# In[104]:


test_x_tfidf


# In[105]:


x_train= data_x_tfidf #the weight of text in train data
y_train = data_y      #the label of text in train data

x_test= test_x_tfidf  #the weight of text in test data
y_test= test_y        #the label of text in test data


# In[106]:


import xgboost as xgb
#build XGboost Classifier
xgb_statement = xgb.XGBClassifier()


# In[107]:


#train XGboost Classifier
xgb_statement.fit(x_train, y_train)


# In[108]:


# predict
y_xgb_statement_pred = xgb_statement.predict(x_test, validate_features=False)
print(y_xgb_statement_pred)


# In[109]:


# ground truth
print(y_test.values)


# In[110]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[111]:


y_true=y_test.values
y_pred=y_xgb_statement_pred
y_true.dtype


# In[112]:


y_pred.dtype


# In[113]:


y_pred = y_pred.astype('int64')


# In[115]:


print('accuracy：{:.2f}%'.format(accuracy_score(y_true, y_pred)*100))
print('precision：{}'.format(precision_score(y_true, y_pred, average=None)))
print('recall：{}'.format(recall_score(y_true, y_pred, average=None)))
print('F-measure：{}'.format(f1_score(y_true, y_pred, average=None)))


# In[116]:


from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
#build LightGBM Classifier
lgb_statement = lgb.LGBMClassifier()


# In[117]:


#turn into float32 for LightGBM
x_train2= data_x_tfidf.astype('float32')
y_train2 = data_y.astype('float32')

x_test2= test_x_tfidf.astype('float32')
y_test2= test_y.astype('float32')


# In[118]:


#the format for feature of LightGBM
lgb_train = lgb.Dataset(x_train2, y_train2)
lgb_val = lgb.Dataset(x_test2, y_test2, reference=lgb_train)


# In[119]:


#set params
params = {'max_depth': 5, 'min_data_in_leaf': 20, 'num_leaves': 35, 'objective':'binary'}
# train LightGBM
gbm = lgb.train(params, lgb_train)


# In[120]:


# predict
y_lgb_statement_pred = gbm.predict(x_test2)
print(y_lgb_statement_pred)
np.where(y_lgb_statement_pred > 0.5,1,0)


# In[121]:


y_true2=y_test2.values
y_pred2=np.where(y_lgb_statement_pred > 0.5,1,0)


# In[122]:


y_true2.dtype


# In[123]:


y_pred2.dtype


# In[124]:


y_true2 = y_true2.astype('int32')


# In[125]:


print('accuracy：{:.2f}%'.format(accuracy_score(y_true2, y_pred2)*100))
print('precision：{}'.format(precision_score(y_true2, y_pred2, average=None)))
print('recall：{}'.format(recall_score(y_true2, y_pred2, average=None)))
print('F-measure：{}'.format(f1_score(y_true2, y_pred2, average=None)))

