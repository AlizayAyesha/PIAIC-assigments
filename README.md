# PIAIC-assigment-2#!/usr/bin/env python
# coding: utf-8


# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[10]:


import numpy as np


# In[26]:


arr = np. array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# In[23]:


arr = np. reshape(arr, (2, 5))


# In[24]:


print(arr)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[115]:


x = [[0, 1, 2, 3, 4] , [5, 6, 7, 8, 9]]


# In[114]:


arr_1 = np.array(x)


# In[113]:


y = [[1, 1, 1, 1, 1] , [1, 1, 1, 1, 1]]


# In[116]:


arr_2 = np.array(y)


# In[117]:


np.vstack((arr_1,arr_2))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[112]:


np.hstack((arr_1,arr_2))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[169]:



print(arr.ndim,"Dimension")
print(arr)
arr = arr.flatten()
print(arr.ndim,"Dimension")
arr


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[200]:


arr = np.arange(15).reshape(-1)
arr


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[197]:


arr = np.arange(15).reshape(-1,3)
arr


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[201]:


arr = np.arange(25).reshape(5,5)
print(arr)
np.square(arr)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[225]:


np.random.seed(123)
arr = np.random.randint(30,size = (5,6))
print(arr)
arr.mean()


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[218]:


np.std(arr)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[222]:


np.median(a)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[226]:


arr.T


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[228]:


arr = np.arange(16).reshape(4,4)
print(arr)
np.diagonal(arr)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[229]:


np.linalg.det(arr)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[230]:


arr = np.arange(10)
print(arr)
print(np.percentile(arr,5))
print(np.percentile(arr,95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[256]:


empty_array= np.array
print(array)
is_empty = (empty_array)
print(is_empty)
