#!/usr/bin/env python
# coding: utf-8

# In[5]:


from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm


# In[6]:


DATADIR = "C:/Users/123/Datasets/STFT/training"


# In[7]:


CATEGORIES = ["STFT0", "STFT1", "STFT2"]


# In[8]:


for category in CATEGORIES:  # do classify green, yellow, red
    path = os.path.join(DATADIR,category)  # create path to green, yellow, red
    for img in os.listdir(path):  # 在每個圖像上重複 green, yellow, red
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
        plt.imshow(img_array)  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!


# In[9]:


cv2.imread("C:/Users/123/Datasets/STFT/training/STFT0/01.PNG")


# In[10]:


cv2.imread("C:/Users/123/Datasets/STFT/training/STFT0/01.PNG").shape


# In[11]:


IMG_SIZE = 64

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


# In[12]:


new_array.shape


# In[13]:


new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array)
plt.show()


# In[14]:


training_data = []

def create_training_data():
    for category in CATEGORIES:  # do classify green, yellow, red

        path = os.path.join(DATADIR,category)  # create path to green, yellow, red
        class_num = CATEGORIES.index(category)  # do the classifiy 0 or 1 or 2, 0=green, 1=yellow, 2=red

        for img in tqdm(os.listdir(path)):  # iterate over each image per green, yellow, red
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


# In[15]:


import random

random.shuffle(training_data)


# In[16]:


for sample in training_data[:10]:
    print(sample[1])


# In[17]:


X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)


# In[18]:


X.shape


# In[19]:


import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[20]:


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)


# In[21]:


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


# In[22]:


y.shape


# In[24]:


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

model.add(Dense(128
         , activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3
               , activation='softmax'))


# In[25]:


print(model.summary())


# In[26]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[27]:


train_history=model.fit(X,
                       y, validation_split=0.2,
                       epochs=10, batch_size=10,verbose=2)


# In[28]:


scores=model.evaluate(X,
                      y)
scores[1]


# In[29]:


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()


# In[30]:


show_train_history(train_history,'acc','val_acc')


# In[31]:


show_train_history(train_history, 'loss','val_loss')


# In[ ]:




