#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


df = pd.read_csv("winequalityN.csv")
df.describe()


# In[3]:


df.hist(bins=25, figsize=(10, 10))
plt.show()


# In[4]:


df.isnull().sum()


# In[5]:


plt.figure(figsize=[10,6], facecolor = 'white')
plt.bar(df['quality'], df['alcohol'], color='green')
plt.xlabel('quality')
plt.ylabel('alcohol')


# In[6]:


sb.displot(x = df['quality'], kde = False, color = 'c')


# In[7]:


#Correlation visualization

plt.figure(figsize=[19,10], facecolor = 'white')
sb.heatmap(df.corr(), annot=True,cmap="YlGnBu")


# In[8]:


feature_name = []
for i in range(len(df.corr().columns)):
    for j in range(i):
        if abs(df.corr().iloc[i,j])>0.7:
            feature_name = df.corr().columns[i]
            print(feature_name)
            


# In[9]:


new_df = df.drop('total sulfur dioxide', axis=1)


# In[10]:


new_df.isnull().sum()


# In[11]:


new_df.update(new_df.fillna(new_df.mean()))


# In[12]:


# For handling categorical variables
cat = new_df.select_dtypes(include='O')

cat_df = pd.get_dummies(new_df, drop_first=True)
cat_df


# In[13]:


# Values greater than 0.7 will consider as 1 and below 0.7 will consider as 0
cat_df['best quality'] = [1 if x>=7 else 0 for x in df.quality]
cat_df


# In[14]:


# Splitting dataset into train and test

from sklearn.model_selection import train_test_split
x = cat_df.drop(['quality','best quality'],axis=1)
y = cat_df['best quality']
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=40)


# In[15]:


# Normalization of numerical variables

from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
norm_fit = norm.fit(x_train)
new_x_train = norm_fit.transform(x_train)
new_x_test = norm_fit.transform(x_test)


# In[16]:


new_x_train


# In[17]:


new_x_test


# In[18]:


# Applying Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import classification_report
random = RandomForestClassifier()
random_fit = random.fit(new_x_train, y_train)
random_score = random.score(new_x_test, y_test)
log = pd.DataFrame(columns=["model", "accuracy"])
log = log.append({"model": "Random Forest", "accuracy": random_score}, ignore_index=True)
# random_score
log


# In[19]:


x_predict = list(random.predict(new_x_test))
df = {'predicted':x_predict,'original':y_test}
pd.DataFrame(df).head(20)


# In[20]:


print(classification_report(x_predict, y_test))


# In[21]:


from sklearn.metrics import accuracy_score
random_forest_accuracy = accuracy_score(y_test,x_predict)
random_forest_accuracy


# In[22]:


# Support Vector Classifier

from sklearn.svm import SVC
model2 = SVC()
model2.fit(new_x_train,y_train)
y_pred = model2.predict(new_x_test)
svc_accuracy = accuracy_score(y_test, y_pred)
svc_accuracy
log = log.append({"model": "Support Vector", "accuracy": svc_accuracy}, ignore_index=True)
log


# In[23]:


# Decision Tree

from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model3.fit(new_x_train,y_train)
y_pred3 = model3.predict(new_x_test)
dt_accuracy = accuracy_score(y_test, y_pred3)
dt_accuracy
log = log.append({"model": "Decision Tree", "accuracy": dt_accuracy}, ignore_index=True)
log


# In[24]:


# SGD Classifier

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(penalty=None)
sgd.fit(new_x_train, y_train)
y_pred4 = sgd.predict(new_x_test)
sgd_accuracy = accuracy_score(y_test, y_pred4)
sgd_accuracy
log = log.append({"model": "SGD ", "accuracy": sgd_accuracy}, ignore_index=True)
log


# In[25]:


# K-NN Classifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(new_x_train, y_train)
y_pred5 = knn.predict(new_x_test)
knn_accuracy = accuracy_score(y_test, y_pred5)
knn_accuracy
log = log.append({"model": "K-Nearest ", "accuracy": knn_accuracy}, ignore_index=True)
log


# In[26]:


# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0).fit(new_x_train, y_train)
y_pred6 = gb.predict(new_x_test)
gb_accuracy = accuracy_score(y_test, y_pred6)
gb_accuracy
log = log.append({"model": "Gradient Boosting ", "accuracy": gb_accuracy}, ignore_index=True)
log


# In[27]:


# Adaboost Classifier

from sklearn.ensemble import AdaBoostClassifier
ada_boost = AdaBoostClassifier(n_estimators=100, random_state=0)
ada_boost.fit(new_x_train, y_train)
y_pred7 = ada_boost.predict(new_x_test)
ada_boost_accuracy = accuracy_score(y_test, y_pred7)
ada_boost_accuracy
log = log.append({"model": "Adaboost  ", "accuracy": ada_boost_accuracy}, ignore_index=True)
log


# In[28]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(new_x_train, y_train)
y_pred8 = lr.predict(new_x_test)
lr_accuracy = accuracy_score(y_test, y_pred8)
lr_accuracy
log = log.append({"model": "Logistic Regression  ", "accuracy": lr_accuracy}, ignore_index=True)
log


# In[29]:


#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(new_x_train, y_train)
y_pred9 = gnb.predict(new_x_test)
gnb_accuracy = accuracy_score(y_test, y_pred9)
# gnb_accuracy
log = log.append({"model": "GaussianNB", "accuracy": gnb_accuracy}, ignore_index=True)
log


# In[ ]:




