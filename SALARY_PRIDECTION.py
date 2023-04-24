#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


#reading csv file
data = pd.read_csv("Salary_dataset.csv")


# In[3]:


# This displays the top 5 rows of the data
data.head()


# In[4]:


# Provides some information regarding the columns in the data
data.info()


# In[5]:


# this describes the basic stat behind the dataset used 
data.describe()


# In[8]:


# These Plots help to explain the values and how they are scattered
plt.scatter(data['YearsExperience'], data['Salary'])
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.show()


# In[7]:


sns.pairplot(data,x_vars=['YearsExperience'], y_vars=['Salary'], size=7,kind='scatter')
plt.show()


# In[9]:


# cooking thr data
X = data['YearsExperience']
X.head()


# In[11]:


Y = data['Salary']
Y.head()


# In[12]:


from sklearn.model_selection import train_test_split


# In[14]:


# Split the data for train and test 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.7,random_state=100)


# In[15]:


# Create new axis for x column
X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]


# In[16]:


# Importing Linear Regression model from scikit learn
from sklearn.linear_model import LinearRegression


# In[17]:


# Fitting the model
lr = LinearRegression()
lr.fit(X_train,Y_train)


# In[18]:


# Predicting the Salary for the Test values
Y_pred = lr.predict(X_test)


# In[21]:


# Plotting the actual and predicted values

c = [i for i in range (1,len(Y_test)+1,1)]
plt.plot(c,Y_test,color='r',linestyle='-')
plt.plot(c,Y_pred,color='b',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()


# In[22]:


# plotting the error
c = [i for i in range(1,len(Y_test)+1,1)]
plt.plot(c,Y_test-Y_pred,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()


# In[23]:


# Importing metrics for the evaluation of the model
from sklearn.metrics import r2_score,mean_squared_error


# In[25]:


# calculate Mean square error
mse = mean_squared_error(Y_test,Y_pred)


# In[26]:


# Calculate R square vale
rsq = r2_score(Y_test,Y_pred)


# In[27]:


print('mean squared error :',mse)
print('r square :',rsq)


# In[29]:


# Just plot actual and predicted values for more insights
plt.figure(figsize=(12,6))
plt.scatter(Y_test,Y_pred,color='r',linestyle='-')
plt.show()


# In[30]:


# Intecept and coeff of the line
print('Intercept of the model:',lr.intercept_)
print('Coefficient of the line:',lr.coef_)


# In[35]:


# visualize the linear regression line
plt.scatter(data['YearsExperience'], data['Salary'])
plt.plot(X_train, lr.predict(X_train), color = 'r')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()


# In[38]:


#       y = 25202.8 + 9731.2x
#     y = mx + b

