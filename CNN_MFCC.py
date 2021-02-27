#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm


# In[2]:


DATADIR = "C:/Users/123/Datasets/MFCC/training"


# In[3]:


CATEGORIES = ["MFCC0", "MFCC1", "MFCC2"]


# In[4]:


for category in CATEGORIES:  # do NAsthma and YAsthma
    path = os.path.join(DATADIR,category)  # create path to NAsthma and YAsthma
    for img in os.listdir(path):  # 在每個圖像上重複 YAsthma and NAsthma
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
        plt.imshow(img_array)  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!


# In[5]:


cv2.imread("C:/Users/123/Datasets/MFCC/training/MFCC0/201.PNG")


# In[6]:


cv2.imread("C:/Users/123/Datasets/MFCC/training/MFCC0/201.PNG").shape


# In[7]:


IMG_SIZE = 64

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


# In[8]:


new_array.shape


# In[9]:


new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array)
plt.show()


# In[10]:


training_data = []

def create_training_data():
    for category in CATEGORIES:  # do NAsthma and YAsthma

        path = os.path.join(DATADIR,category)  # create path to NAsthma and YAsthma
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=NAsthma 1=YAsthma

        for img in tqdm(os.listdir(path)):  # iterate over each image per NAsthma and YAsthma
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
            


create_training_data()



print(len(training_data))


# In[11]:


import random

random.shuffle(training_data)


# In[12]:


for sample in training_data[:10]:
    print(sample[1])


# In[13]:


X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)


# In[14]:


X.shape


# In[15]:


import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[16]:


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


# In[17]:


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0
y=np_utils.to_categorical(y)


# In[18]:


y.shape


# In[19]:


#Add CNN model
model = Sequential()

model.add(Conv2D(filters=16
                 , kernel_size=(3, 3)
                 , padding="same"
                 , input_shape=(64, 64, 3)
                 , activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64
                 , kernel_size=(3, 3)
                 , padding="same"
                 , activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(256
         , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3
               , activation='softmax'))


# In[20]:


print(model.summary())


# In[21]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[63]:


train_history=model.fit(X,
                       y, validation_split=0.2,
                       epochs=10, batch_size=10,verbose=2)


# In[64]:


scores=model.evaluate(X,
                      y)
scores[1]


# In[65]:


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()


# In[66]:


show_train_history(train_history,'acc','val_acc')


# In[67]:


show_train_history(train_history, 'loss','val_loss')


# In[68]:


prediction=model.predict_classes(X)


# In[69]:


prediction[:5]


# In[70]:


#from tensorflow.keras.preprocessing import image

#dir_path = 'C:/Users/123/Datasets/MFCC/testing'

#for i in os.listdir(dir_path ):

   # img = image.load_img(dir_path+'//'+ i, target_size=(64, 64))
    #IMG_SIZE = 64
    #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
   # plt.imshow(img)
   # plt.show()
    

   
    #img2=cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #X = image.img_to_array(img)
    #X = np.expand_dims(X, axis = 0)
    #images = np.vstack([X])
    #val= model.predict(images)
    #if val == 0 :
     #   print ("You are great")
    #else:
     #   print ("You got Asthma")


# In[ ]:




