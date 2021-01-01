#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[4]:


import numpy as np


# 2. Create a null vector of size 10 

# In[43]:


null_vector = np.zeros(10)
null_vector 


# 3. Create a vector with values ranging from 10 to 49

# In[9]:


arr = np.arange(10,49)
arr


# 4. Find the shape of previous array in question 3

# In[11]:


np.shape(arr)


# 5. Print the type of the previous array in question 3

# In[12]:


type(arr)


# 6. Print the numpy version and the configuration
# 

# In[13]:


np.__version__


# In[14]:


np.show_config()


# 7. Print the dimension of the array in question 3
# 

# In[ ]:





# 8. Create a boolean array with all the True values

# In[16]:


arr = np.ones(10,dtype = bool)
arr


# 9. Create a two dimensional array
# 
# 
# 

# In[26]:


arr = np.ones(10)
arr


# In[27]:


arr.ndim


# In[28]:


arr = arr.reshape(2,5)
arr


# In[29]:


arr.ndim


# 10. Create a three dimensional array
# 
# 

# In[54]:


arr = np.arange(1,13)
arr


# In[31]:


arr.ndim


# In[34]:


arr = arr.reshape(2,2,3)


# In[35]:


arr.ndim


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[36]:


arr = np.arange(1,10)
arr


# In[37]:


arr[::-1]


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[44]:


null_vector = np.zeros(10)
null_vector[4]=1
null_vector


# 13. Create a 3x3 identity matrix

# In[45]:


identity = np.identity(3)
identity


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[73]:


arr = np.array([1,2,3,4,5,6,7,8,9,10])
print (arr)
print(arr.dtype)
arr = arr.astype("float64")
print(arr)
print(arr.dtype)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[98]:


arr1 = np.array([[1, 2, 3, 4, 5 ] , [6, 7, 8, 9, 10]])
arr2 = np.array([[1., 2., 3., 4., 5.] , [6., 7., 8., 9., 10.]])
multiply = arr1*arr2
multiply


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[106]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
maxi = np.maximum(arr1,arr2)
maxi


# 17. Extract all odd numbers from arr with values(0-9)

# In[108]:


arr = np.arange(1,10)
arr


# In[109]:


arr[1::2]


# 18. Replace all odd numbers to -1 from previous array

# In[113]:


arr[1::2] = -1
arr


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[114]:


arr = np.arange(10)
arr


# In[115]:


arr[5:-1] = 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[166]:


b = np.arange(12).reshape(4,3)
b[0,0]=1
b[0,2]=1
b[1,0]=1
b[1,2]=1
b[2,0]=1
b[2,2]=1
b[3,0]=1
b[3,2]=1
b[3,1]=1
b[1,1]=0
b[2,1]=0
b


# # Dificuilty level Medium

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[121]:


arr_2d =np.array([[1, 2, 3] , [4, 5, 6] , [7, 8, 9]])
arr_2d[1,1] =12
arr_2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[239]:


arr3d = np.arange(12).reshape(4,3)
arr3d[2:]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[224]:


b = np.arange(9).reshape(3,3)
b[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[263]:


arr = np.arange(0,9).reshape(3,3)
arr


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[264]:


arr[0]


# In[265]:


arr[1]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[255]:


arr = np.random.randint(100,size=(10,10))
arr


# In[256]:


print (np.min(arr))
print (np.max(arr))


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[261]:


a = np.array([1,2,3,4,5,6])
b = ([7,2,10,2,7,4,9,4,9,8])
R = np.intersect1d(a,b)
R


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[270]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7,4)
data[names != "Wil"]


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[327]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(data[names !="Will"])


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[272]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(data[names !="Will"])
print(data[names !="Joe"])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[281]:


arr = np.random.randn(1,15).reshape(5,3)
arr


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[329]:


dara = np.random.randn(1,16).reshape(2,2,4)
dara


# 33. Swap axes of the array you created in Question 32

# In[330]:


data.T


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[305]:


r = np.arange(10)
r = np.sqrt(r)
r = np.where(r<0.5,0,r)
r


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[309]:


a = np.random.randint(12)
b = np.random.randint(12)
np.maximum(a,b)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[311]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
names = set(names)
names


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[312]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
a[b[np.searchsorted(b,a)] !=  a]


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[317]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray[2:] = ([[10,10,10]])
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[322]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[325]:


a = np.random.randint(20,size=(4,4))
a


# In[326]:


np.sum(a)

