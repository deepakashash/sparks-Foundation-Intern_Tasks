#!/usr/bin/env python
# coding: utf-8

# ## To Explore Supervised Machine Learning
# 
# # Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # loading the data

# In[2]:


url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")


# In[3]:


s_data.head(20)


# 
# # Plotting the distribution of scores

# In[4]:


s_data.plot(x="Hours", y= "Scores", style="o")
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage score')
plt.show()


# ###  From the above graph we can see clearly that there is a positive relationship between number of hours studied and percentage of scor

# #  Preparing the data

# ## The next step is to divide the data into "attributes"(inputs) and "labels"(outputs)

# In[5]:


X = s_data.iloc[:, :-1].values
y = s_data.iloc[:, 1].values


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Training the Algorithm

# In[7]:



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training Complete")


# In[8]:



### Plotting the regression line
line = regressor.coef_*X+regressor.intercept_


### Plotting for the test data
plt.scatter(X,y)
plt.plot(X,line);
plt.show()


# # Making Predictions

# In[9]:


print(X_test)     ### Testing data in Hours
y_pred = regressor.predict(X_test)  ### Predicting the scores


# In[10]:


### Comparing Actual value vs Predicted Value
df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df


# ## Testing with own Data

# In[11]:


Hours = np.array([[9.25]])
own_pred=regressor.predict(Hours)
print("No.of hours = {}".format(Hours))
print("Predicted Score = {}".format(own_pred[0]))


# ## Evaluating the Model

# In[12]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))


# In[13]:


print('Mean Squared Error:',metrics.mean_squared_error(y_test, y_pred))


# In[14]:


print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




