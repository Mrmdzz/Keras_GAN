
# In[1]:
ComputeLB = True
DogsOnly = True
import cv2
import numpy as np, pandas as pd, os
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt, zipfile 
from PIL import Image 
import os
import cv2
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import albumentations as albu
from albumentations import OneOf
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow_core.python.keras.activations import relu,tanh,linear
from tensorflow.keras.layers import *
#from tensorflow.keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
#from network.custom_layers import PixelShuffler, Scale
import keras.backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply,Softmax,Lambda,add,BatchNormalization
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import tensorflow_core
from tensorflow.keras.callbacks import LearningRateScheduler
#from tta_wrapper import tta_segmentation
#from keras_radam.training import RAdamOptimizer
#from keras_radam import RAdam
#from keras.UNetPlusPlus.segmentation_models import Xnet,Unet
import segmentation_models as sm
#from tensorflow.python.keras.models import load_model
print(tf.__version__)
from tensorflow.keras.models import load_model,save_model


# In[2]:


train = pd.read_csv('E:/Competition_data/Understanding_Clouds_from_Satellite_Images/understanding_cloud_organization/train.csv')
submission = pd.read_csv('E:/Competition_data/Understanding_Clouds_from_Satellite_Images/understanding_cloud_organization/sample_submission.csv')
print('Number of train samples:', train.shape[0])
print('Number of test samples:', submission.shape[0])

# Preprocecss data
train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
test = pd.DataFrame(submission['image'].unique(), columns=['image'])

# #### With mask

# In[4]:
# ### Let's take a look at a sample from each class

# In[6]:

train['Has Mask'] = ~train['EncodedPixels'].isna()
maskedSamples = train[train['Has Mask'] == True]
firstLabel = maskedSamples.groupby('label').first().reset_index()

# In[9]:


maskedSamples_gp = maskedSamples.groupby('image').size().reset_index(name='Number of masks')

# ## Split train and validation sets

# In[10]:


mask_count_df = train.groupby('image').agg(np.sum).reset_index()
mask_count_df.sort_values('Has Mask', ascending=False, inplace=True)
#train_idx, val_idx = train_test_split(mask_count_df.index, test_size=0.2, random_state=seed)


# In[11]:


def np_resize(img, input_shape):
    height, width = input_shape
    return cv2.resize(img, (width, height))
    
def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(rle, input_shape):
    width, height = input_shape[:2]
    
    mask = np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T

def build_masks(rles, input_shape, reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape, depth))
    else:
        masks = np.zeros((*reshape, depth))
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle2mask(rle, input_shape)
            else:
                mask = rle2mask(rle, input_shape)
                reshaped_mask = np_resize(mask, reshape)
                masks[:, :, i] = reshaped_mask
    
    return masks

def build_rles(masks, reshape=None):
    width, height, depth = masks.shape
    rles = []
    
    for i in range(depth):
        mask = masks[:, :, i]
        
        if reshape:
            mask = mask.astype(np.float32)
            mask = np_resize(mask, reshape).astype(np.int32)
        
        rle = mask2rle(mask)
        rles.append(rle)
        
    return rles
    
BATCH_SIZE =2
EPOCHS = 60
LEARNING_RATE = 3e-4
HEIGHT =350
WIDTH = 525
CHANNELS = 3
N_CLASSES = train['label'].nunique()
print("有",N_CLASSES,"個類")
ES_PATIENCE = 5
RLROP_PATIENCE = 3
# ## Data generator
# 
# #### I got the data generators and predictions from @xhlulu kernel: [Satellite Clouds: Yet another U-Net boilerplate](https://www.kaggle.com/xhlulu/satellite-clouds-yet-another-u-net-boilerplate/notebook) check out, I just changed a few things to make the code more familiar to me.

# In[13]:


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='E:/Competition_data/Understanding_Clouds_from_Satellite_Images/understanding_cloud_organization/train_images/',
                 batch_size=BATCH_SIZE, dim=(1400, 2100), n_channels=CHANNELS, reshape=None, 
                 n_classes=N_CLASSES, random_state=1024, shuffle=False, augment=False):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.reshape = reshape
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.random_state = random_state
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        X,Xname = self.__generate_X(list_IDs_batch)
        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            
            if self.augment:
                X, y = self.__augment_batch(X, y)
            
            return X,y,Xname#[X,X,X,X,X,X], [y,y,y,y,y,y]
        
        elif self.mode == 'predict':
            return X#[X,X,X,X,X,X]

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        """if self.shuffle == False:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)"""
    
    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        if self.reshape is None:
            X = np.empty((self.batch_size, *self.dim,self.n_channels))
            Xname=[]
        else:
            X = np.empty((self.batch_size, *self.reshape,self.n_channels))
            Xname=[]
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['image'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.
            
            if self.reshape is not None:
                img = np_resize(img, self.reshape)
            
            # Store samples
            X[i,] = img
            Xname.append(im_name)
        return X.astype(np.float32),Xname
    
    def __generate_y(self, list_IDs_batch):
        if self.reshape is None:
            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        else:
            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['image'].iloc[ID]
            image_df = self.target_df[self.target_df['image'] == im_name]
            
            rles = image_df['EncodedPixels'].values
            
            if self.reshape is not None:
                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)
            else:
                masks = build_masks(rles, input_shape=self.dim)
            
            y[i, ] = masks

        return y.astype(np.float32)
    
    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ], masks_batch[i, ] = self.__random_transform(img_batch[i, ], masks_batch[i, ])
        
        return img_batch, masks_batch
    
    def __random_transform(self, img, masks):
        composition = albu.Compose([albu.HorizontalFlip(),
                               albu.VerticalFlip(),
                               albu.ShiftScaleRotate(),
                               
                               #albu.RandomSizedCrop(min_max_height=(264, 352), height=352, width=512,p=0.25),
                               OneOf([
                                       albu.RandomContrast(),
                                       albu.RandomGamma(),
                                       albu.RandomBrightness(),
                                       ], p=0.25),
        OneOf([
                albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                albu.GridDistortion(),
                albu.OpticalDistortion(distort_limit=2, shift_limit=0.25),
        ], p=0.25),
                              ])
        
        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        
        return aug_img, aug_masks

# In[7]:
def Valid_image(dataset,valid_indexes,batch_size,flag=True):
    X_train=[]
    y_train=[]
    Z_train=[]
    for each in range(0, valid_indexes):
        for x in range(0,batch_size):
            X_train.append((np.array(dataset[each][0][x])).reshape(( HEIGHT, WIDTH, 3)))
            y_train.append((np.array(dataset[each][1][x])).reshape(( HEIGHT, WIDTH, 4)))
            Z_train.append((np.array(dataset[each][2][x])))
    return [X_train,y_train,Z_train]
#train_indexes, valid_indexes = train_test_split(mask_count_df.index, test_size=0.01, random_state=None)

train_generator = DataGenerator(
                  mask_count_df.index[0:40], 
                  df=mask_count_df,
                  target_df=train,
                  batch_size=BATCH_SIZE,
                  reshape=(HEIGHT, WIDTH),
                  n_channels=CHANNELS,
                  n_classes=N_CLASSES,
                  random_state=1024)
train=Valid_image(train_generator,len(train_generator),batch_size=BATCH_SIZE,flag=True)
Lsize=len(train[0])
train_y=np.array(train[0])
train_z=train[2]
# # Build Discriminator

# In[2]:

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (np.abs(data - np.min(data))+1e-9) / (_range+1e-9)
# In[3]:

# BUILD DISCRIMINATIVE NETWORK
dog = Input((HEIGHT*WIDTH*3,))
dogName = Input((Lsize,))
x = Dense(HEIGHT*WIDTH*3, activation='sigmoid')(dogName) 
x = Reshape((2,HEIGHT*WIDTH*3,1))(concatenate([dog,x]))
x = Conv2D(1,(2,1),use_bias=False,name='conv')(x)
discriminated = Flatten()(x)

# COMPILE
discriminator = Model([dog,dogName], discriminated)
discriminator.get_layer('conv').trainable = False
discriminator.get_layer('conv').set_weights([np.array([[[[-1.0 ]]],[[[1.0]]]])])
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# DISPLAY ARCHITECTURE
discriminator.summary()


# # Train Discriminator
# We will train the Discriminator to memorize the training images. (Typically you don't train the Discriminator ahead of time. The D learns as the G learns. But this GAN is special).

# In[4]:

# TRAINING DATA
train_y = np.array(train_y).reshape((-1,HEIGHT*WIDTH*3))
print(np.array(train_y).shape)
train_X = np.zeros((Lsize,Lsize))
for i in range(Lsize): train_X[i,i] = 1
zeros = np.zeros((Lsize,HEIGHT*WIDTH*3))
# TRAIN NETWORK
lr = 0.5
for k in range(10):
    annealer = LearningRateScheduler(lambda x: lr)
    h = discriminator.fit([zeros,train_X ], train_y, epochs = 10, batch_size=BATCH_SIZE, callbacks=[annealer], verbose=0)
    print('Epoch',(k+1)*10,'/30 - loss =',h.history['loss'][-1] )
    if h.history['loss'][-1]<0.533: lr = 0.1


# # Delete Training Images
# Our Discriminator has memorized all the training images. We will now delete the training images. Our Generator will never see the training images. It will only be coached by the Discriminator. Below are examples of images that the Discriminator memorized.

# In[5]:


del train_X, train_y


# In[6]:


print('Discriminator Recalls from Memory Dogs')    
for k in range(5):
    plt.figure(figsize=(15,3))
    for j in range(5):
        xx = np.zeros((Lsize))
        xx[np.random.randint(Lsize)] = 1
        plt.subplot(1,5,j+1)
        img = discriminator.predict([zeros[0,:].reshape((-1,HEIGHT*WIDTH*3)),xx.reshape((-1,Lsize))]).reshape((-1,HEIGHT, WIDTH,3))
        img = normalization(img)
        img = Image.fromarray( (255*img).astype('uint8').reshape((HEIGHT, WIDTH,3)))
        plt.axis('off')
        plt.imshow(img)
    plt.show()
# In[7]:
# BUILD GENERATOR NETWORK
BadMemory = False

if BadMemory:
    seed = Input((Lsize,))
    x = Dense(2048, activation='tanh')(seed)
    x = Reshape((8,8,32))(x)
    x = Conv2D(128, (3, 3), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='linear', padding='same')(x)
    generated = Flatten()(x)
else:
    seed = Input((Lsize,))
    generated = Dense(HEIGHT*WIDTH*3,activation='elu', kernel_initializer='he_normal')(seed)
# COMPILE
generator = Model(seed, [generated,Reshape((Lsize,))(seed)])

# DISPLAY ARCHITECTURE
generator.summary()


# In[8]:


# BUILD GENERATIVE ADVERSARIAL NETWORK
   
gan_input = Input(shape=(Lsize,))
x = generator(gan_input)
discriminator.trainable=False 
gan_output = discriminator(x)

# COMPILE GAN
gan = Model(gan_input, gan_output)

gan.get_layer(index=2).get_layer('conv').set_weights([np.array([[[[-1 ]]],[[[255.]]]])])
gan.compile(optimizer='adam', loss='mean_squared_error')

# DISPLAY ARCHITECTURE
gan.summary()


# # Discriminator Coaches Generator
# In a typical GAN, the discriminator does not memorize the training images beforehand. Instead it learns to distinquish real images from fake images at the same time that the Generator learns to make fake images. In this GAN, we taught the Discriminator ahead of time and it will now teach the Generator.

# In[9]:


# TRAINING DATA
train = np.zeros((Lsize,Lsize))#np.random.uniform(0, 1, size=(Lsize,Lsize))
for i in range(Lsize): train[i,i] = 1
zeros = np.ones((Lsize,HEIGHT*WIDTH*3))

# TRAIN NETWORKS
ep = 1; it = 30
if BadMemory: lr = 0.005
else: lr = 2.  
for k in range(it):  

    # BEGIN DISCRIMINATOR COACHES GENERATOR
    annealer = LearningRateScheduler(lambda x: lr)
    h = gan.fit(train, zeros, epochs = ep, batch_size=BATCH_SIZE, callbacks=[annealer], verbose=0)

    # DISPLAY GENERATOR LEARNING PROGRESS 
    print('Epoch',(k+1),'/'+str(it)+' - loss =',h.history['loss'][-1] )
    plt.figure(figsize=(15,3))
    for j in range(5):
        xx = np.zeros((Lsize))
        xx[np.random.randint(Lsize)] = 1
        plt.subplot(1,5,j+1)
        img = generator.predict(xx.reshape((-1,Lsize)))[0].reshape((-1,HEIGHT,WIDTH,3))
        img = normalization(img)
        img = Image.fromarray( (255.0*img).astype('uint8').reshape((HEIGHT,WIDTH,3)))
        plt.axis('off')
        plt.imshow(img)
    plt.show()  
            
    # ADJUST LEARNING RATES
    if BadMemory:
        ep *= 2
        if ep>=32: lr = 0.001
        if ep>256: ep = 256
    else:
        if h.history['loss'][-1] < 300: lr = 1.
        if h.history['loss'][-1] < 150: lr = 0.5
        if h.history['loss'][-1] < 130: lr = 0.01
        if h.history['loss'][-1] < 128: lr = 0.005
        if h.history['loss'][-1] < 125: lr = 0.001
        if h.history['loss'][-1] < 110: lr = 0.0005
        if h.history['loss'][-1] < 105: lr = 0.00025
        if h.history['loss'][-1] < 100: lr = 0.0001

# # Build Generator Class
# Our Generative Network has now learned all the training images from our Discriminative Network. With its poor memory, we hope that it has learned to generalize somewhat. Now let's build a Generator Class that accepts any random 100 dimensional vector and outputs an image. Our class will return 70% of one "memorized" image mixed with 30% another. Since the images are stored in a convolutional network, we hope that it makes a generalized conceptual mixture (versus a pixel blend).

# In[10]:


class DogGenerator:
    index = 0   
    def getDog(self,seed):
        xx = np.zeros((Lsize))
        xx[self.index] = 0.70
        xx[np.random.randint(Lsize)] = 0.30
        img = generator.predict(xx.reshape((-1,Lsize)))[0].reshape((HEIGHT,WIDTH,3))
        self.index = (self.index+1)%Lsize
        return Image.fromarray( img.astype('uint8') ) 


# # Examples of Generated Dogs

# In[11]:


# DISPLAY EXAMPLE DOGS
d = DogGenerator()
for k in range(3):
    plt.figure(figsize=(15,3))
    for j in range(5):
        plt.subplot(1,5,j+1)
        img = d.getDog(np.random.normal(0,1,100))
        plt.axis('off')
        plt.imshow(img)
    plt.show() 


# # Submit to Kaggle
# In this kernel we learned how to make an experimental GAN. Currently it scores around LB 100. We must be careful as we try to improve its score. If we give this GAN excellent memory and request a mixture of 99.9% one image and 0.1% another, then it can score LB 7 but then it is returning "altered versions" of images and violates the rules [here][1]
# 
# [1]: https://www.kaggle.com/c/generative-dog-images/discussion/98183

# In[12]:


# SAVE TO ZIP FILE NAMED IMAGES.ZIP
z = zipfile.PyZipFile('E:/working/images.zip', mode='w')
d = DogGenerator()
for k in range(len(train[2])):
    img = d.getDog(np.random.normal(0,1,100))
    f = str(train_z[k])
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(np.array(img),(2100,1400),interpolation=cv2.INTER_CUBIC)
    img = cv2.imwrite('E:/working/'+f,img)
    z.write(f); os.remove(f)
    #img.save(f,'Jpeg'); z.write(f); os.remove(f)
    #if k % 1000==0: print(k)
z.close()
