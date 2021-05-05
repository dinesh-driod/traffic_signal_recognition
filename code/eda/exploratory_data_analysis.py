# importing essential libraries
import warnings
warnings.filterwarnings('ignore')
# from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import seaborn as sns
import cv2
import os
from PIL import ImageEnhance, Image


#Train.csv contains information regarding each train image
df_train = pd.read_csv('/home/ubuntu/capstone_project/data/Train.csv')
labels_train = df_train['ClassId']

#Test.csv contains information regarding each test image
df_test = pd.read_csv('/home/ubuntu/capstone_project/data/Test.csv')
labels_test = df_test['ClassId']
print(df_train.head())

print("No of images in train data : {0}".format(df_train.shape[0]))
print("No of images in test data : {0}".format(df_test.shape[0]))

#Checking class distribution in train data
fig, ax = plt.subplots(1, 1)
fig.set_size_inches((25,10))
sns.countplot(data = df_train, x = 'ClassId', )
plt.title('Class Distribution in train data',fontsize=16)
plt.savefig('train_distribution.png')

'''
Observations :
The target classes are clearly not uniformly distributed.
This is quite logical since some signs like "Keep speed below 30Kmph" or "A bump ahead" appears more often then signs like "Road under construction ahead".
Also, for most of the classes, the imbalance is not severe.
'''

#Checking class distribution in test data
fig, ax = plt.subplots(1, 1)
fig.set_size_inches((25,10))
sns.countplot(data = df_test, x = 'ClassId', )
plt.title('Class Distribution in test data',fontsize=16)
plt.savefig('test_distribution.png')
plt.show()
'''
Observations :
The observations for test data are quite similer to that of the train data.
Here also some classes are more frequent as compared to the others.
'''

#Distplot for Height and Width of images in train data
fig, ax = plt.subplots(2, 2)
fig.set_size_inches((20,12))
sns.distplot(df_train['Height'], ax = ax[0][0])
fig.set_size_inches((20,12))
sns.distplot(df_test['Height'], ax = ax[0][1])
fig.set_size_inches((20,12))
sns.distplot(df_train['Width'], ax = ax[1][0])
fig.set_size_inches((20,12))
sns.distplot(df_test['Width'], ax = ax[1][1])
plt.savefig('Density_distribution.png')
plt.show()
'''
Observations :
It is evident from the above figures that both our train and test set images follow similer Height and Width distribution.
Since all the images have different dimensions, we need to fix the height and width constant for every image to follow.
We need to perform this in such a way that data loss is least.
For smaller images, we will need to do appropriate padding.
'''
def load_train_images() :
  '''
  This functions loads train images into a numpy
  array and returns it. It also returns the train
  labels.
  '''
  #Reading the input images and storing then into numpy.ndarray
  n_classes = 43
  data = []
  labels = []
  #Reading the train images
  for itr in range(n_classes) :
    path = '/home/ubuntu/capstone_project/data/Train/{0}/'.format(itr)
    Class = os.listdir(path)
    for c in Class :
      image = cv2.imread(path + c)
      image = Image.fromarray(image)
      data.append(np.array(image))
      labels.append(itr)
  data=np.array(data)
  labels=np.array(labels)
  return data, labels

def img_dims_donut_chart(data) :
  '''
  This function takes images in the form of arrays
  as input and plots a donut chart to show what
  proportion of these images lies within certain dimension range.
  '''
  img_size_dict = {
    "b/w_16_&_32" : 0,
    "b/w_32_&_64" : 0,
    "b/w_64_&_128" : 0,
    "b/w_128_&_256" : 0,
    }
  for img in data :
    if (img.shape[0] > 16 and img.shape[1] > 16) and (img.shape[0] < 32 and img.shape[1] < 32) :
      img_size_dict['b/w_16_&_32'] += 1
    elif (img.shape[0] > 32 and img.shape[1] > 32) and (img.shape[0] < 64 and img.shape[1] < 64) :
      img_size_dict['b/w_32_&_64'] += 1
    elif (img.shape[0] > 64 and img.shape[1] > 64) and (img.shape[0] < 128 and img.shape[1] < 128) :
      img_size_dict['b/w_64_&_128'] += 1
    elif (img.shape[0] > 128 and img.shape[1] > 128) and (img.shape[0] < 256 and img.shape[1] < 256) :
      img_size_dict['b/w_128_&_256'] += 1
  categories = ["Between 16 & 32", "Between 32 & 64", "Between 64 & 128", "Between 128 & 256"]
  sizes = [4766, 23232, 5841, 518]
  my_circle=plt.Circle( (0,0), 0.6, color='white')
  fig = plt.figure(figsize=(25,10))
  plt.pie(sizes, labels=categories, colors=['red','green','blue','skyblue'], shadow=True)
  plt.title("Size range of train images")
  p=plt.gcf()
  p.gca().add_artist(my_circle)
  p.savefig('Doughnut_distribution.png')
  fig.savefig('Doughnut_distribution.png')
  fig.show()
#Loading train images in numpy.ndarray
data, labels = load_train_images()
img_dims_donut_chart(data)

'''
From the above plots, we can conclude that there are very few images that has width and height more than 128.
Hence, we can safely set the size of each image to be 128*128. This won't lead to much data loss.
However, we do need to take care of appropriate padding. Padding will not be same for each image. An image with white background cannot have a black padding.
Let's perform padding on an image and plot them.
We will use copyMakeBorder module of OpenCV.
'''
