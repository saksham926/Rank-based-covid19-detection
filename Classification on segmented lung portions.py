import os,glob
import numpy as np
import os
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import cv2
gpu=input("Which gpu number you would like to allocate:") #TO SELECT GPU
os.environ["CUDA_VISIBLE_DEVICES"]=gpu
model_name=int(input("Which model you would like to train(TYPE THE NUMBER ONLY LIKE 1,2,3)? 1. Densenet 201    2. SE Inception v3   3. SE SQUEEZENET"))
import glob
import pickle
import tensorflow as tf
import argparse
import re
import datetime
import keras
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,Layer,ReLU, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from classification_models.keras import Classifiers
# from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from skimage.transform import radon, rescale
from skimage.filters import roberts, sobel, scharr, prewitt
from classification_models.keras import Classifiers
from skimage import feature
import os,glob
import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
import pickle
import argparse
import re
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from tensorflow.keras.metrics import Recall, Precision
from skimage import data, exposure
from tensorflow.keras.layers import Layer
from PIL import Image
from numpy import asarray
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf

def no_data_augmentation(normal_files,covid_files,pneumonia_files):
    aug_normal=[]
    aug_covid=[]
    aug_pneumonia=[]
    for ele in normal_files:
        #ele=ele/255.0
        x = np.load(ele)
        
        
        
        
        aug_normal.append(x)
    for ele in covid_files:
        #ele=ele/255.0
        x = np.load(ele)
        
        
        
        aug_covid.append(x)
    for ele in pneumonia_files:
        #ele=ele/255.0
        x = np.load(ele)
      
      
        aug_pneumonia.append(x)
    
    for i in range(len(aug_normal)):
        aug_normal[i]=aug_normal[i].reshape((224,224))
    
    for i in range(len(aug_covid)):
        aug_covid[i]=aug_covid[i].reshape((224,224))
    for i in range(len(aug_pneumonia)):
        aug_pneumonia[i]=aug_pneumonia[i].reshape((224,224))
    print("Normal files without augmentation:",len(aug_normal))
    print("Covid files without augmentation:", len(aug_covid))
    print("Pneumonia files without augmentation:",len(aug_pneumonia))
    return aug_normal,aug_covid,aug_pneumonia

def data_augmentation(normal_files,covid_files,pneumonia_files):
    aug_normal=[]
    aug_covid=[]
    thresh_hold=7
    aug_pneumonia=[]
    
    #x = tf.keras.preprocessing.image.load_img("/content/IM-0001-0001.jpeg")
    
    datagen=ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,

    )
    #normal
    counter=0
    
    for location in tqdm(normal_files):
        counter=0

        x = np.load(location)
       
        
        x = np.expand_dims(x, axis=-1) 
        x=x.reshape((1,)+x.shape)
        #x=x/255.0
        

        for i in datagen.flow(x):
            if counter>=17:
                break
            #i=i/255.0

            #i = cv2.resize(i,(224,224),interpolation = cv2.INTER_CUBIC)
            aug_normal.append(i)
            counter+=1
    #covid
    counter=0
    for location in tqdm(covid_files):
        counter=0
        x = np.load(location)
    
        
        x = np.expand_dims(x, axis=-1) 
        #x=img_to_array(x)
        x=x.reshape((1,)+x.shape)
        #x=x/255.0


        for i in datagen.flow(x):
            if counter>=2:
                break

            aug_covid.append(i)
            counter+=1    
    #pneumonia
    counter=0
    for location in tqdm(pneumonia_files):
        counter=0
        x = np.load(location)
    

      
        x = np.expand_dims(x, axis=-1) 
        #x=img_to_array(x)
        x=x.reshape((1,)+x.shape)
        #x=x/255.0

        for i in datagen.flow(x):
            if counter>=3:
                break
            #i=i/255.0
            #i = cv2.resize(i,(224,224),interpolation = cv2.INTER_CUBIC)
            aug_pneumonia.append(i)
            counter+=1    

    for ele in normal_files:
        #ele=ele/255.0
        x = np.load(ele)
        
       
        aug_normal.append(x)
    for ele in covid_files:
        #ele=ele/255.0
        x = np.load(ele)
        
      
        aug_covid.append(x)
    for ele in pneumonia_files:
        #ele=ele/255.0
        x = np.load(ele)
      
      
        aug_pneumonia.append(x)
   
    for i in range(len(aug_normal)):
        aug_normal[i]=aug_normal[i].reshape((224,224))
    
    for i in range(len(aug_covid)):
        aug_covid[i]=aug_covid[i].reshape((224,224))
    for i in range(len(aug_pneumonia)):
        aug_pneumonia[i]=aug_pneumonia[i].reshape((224,224))
    print("Normal files after augmentation:",np.shape(np.array(aug_normal)))
    print("Covid files after augmentation:", np.shape(np.array(aug_covid)))
    print("Pneumonia files after augmentation:",np.shape(np.array(aug_pneumonia)))
    return aug_normal,aug_covid,aug_pneumonia

def making_full_data(aug_normal,aug_covid,aug_pneumonia):
    aug_normal=shuffle(aug_normal, random_state=0)
    aug_covid=shuffle(aug_covid,random_state=0)
    aug_pneumonia=shuffle(aug_pneumonia,random_state=0)
    
    aug_normal_labels=[]
    for i in range(len(aug_normal)):
        aug_normal_labels.append(0)
    print(np.shape(aug_normal),np.shape(aug_normal_labels))
    aug_covid_labels=[]
    for i in range(len(aug_covid)):
        aug_covid_labels.append(1)
    print(np.shape(aug_covid),np.shape(aug_covid_labels))
    aug_pneumonia_labels=[]
    for i in range(len(aug_pneumonia)):
        aug_pneumonia_labels.append(2)
    print(np.shape(aug_pneumonia),np.shape(aug_pneumonia_labels))  

    full_data=[]
    full_label=[]
    for i in range(len(aug_normal)):
        full_data.append(aug_normal[i])
        full_label.append(aug_normal_labels[i])
    for i in range(len(aug_covid)):
        full_data.append(aug_covid[i])
        full_label.append(aug_covid_labels[i])
    for i in range(len(aug_pneumonia)):
        full_data.append(aug_pneumonia[i])
        full_label.append(aug_pneumonia_labels[i])
        
    full_data=np.array(full_data)
    full_label=np.array(full_label)
    
    full_data=shuffle(full_data,random_state=0)
    full_label=shuffle(full_label,random_state=0)
    
    return full_data,full_label
"""Inception 2D_CNN Models in Tensorflow-Keras.
References -
Inception_v1 (GoogLeNet): https://arxiv.org/abs/1409.4842 [Going Deeper with Convolutions]
Inception_v2: http://arxiv.org/abs/1512.00567 [Rethinking the Inception Architecture for Computer Vision]
Inception_v3: http://arxiv.org/abs/1512.00567 [Rethinking the Inception Architecture for Computer Vision]
Inception_v4: https://arxiv.org/abs/1602.07261 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning]
"""




def Conv_2D_Block(x, model_width, kernel, strides=(1, 1), padding="same"):
    # 2D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)
    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs         : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)
    return out


def SE_Block(inputs, num_filters, ratio):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    excitation = tf.keras.layers.Dense(units=num_filters/ratio)(squeeze)
    excitation = tf.keras.layers.Activation('relu')(excitation)
    excitation = tf.keras.layers.Dense(units=num_filters)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([1, 1, num_filters])(excitation)

    scale = inputs * excitation

    return scale


def Inceptionv1_Module(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1), padding='valid')

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1), padding='valid')
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 3))

    branch5x5 = Conv_2D_Block(inputs, filterB3_1, (1, 1), padding='valid')
    branch5x5 = Conv_2D_Block(branch5x5, filterB3_2, (5, 5))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))
    out = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_Block_'+str(i))

    return out


def Inceptionv2_Module(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 3))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 3))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_'+str(i))

    return out


def Inception_Module_A(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch5x5 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch5x5 = Conv_2D_Block(branch5x5, filterB2_2, (5, 5))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 3))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_A'+str(i))

    return out


def Inception_Module_B(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch7x7 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (1, 7))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (7, 1))

    branch7x7dbl = Conv_2D_Block(inputs, filterB3_1, 1)
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_2, (1, 7))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_2, (7, 1))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_3, (1, 7))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_3, (7, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='Inception_Block_B'+str(i))

    return out


def Inception_Module_C(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3_2 = Conv_2D_Block(branch3x3, filterB2_2, (1, 3))
    branch3x3_3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 1))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (1, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 1))
    branch3x3dbl_2 = Conv_2D_Block(branch3x3dbl, filterB3_3, (1, 3))
    branch3x3dbl_3 = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch3x3_2, branch3x3_3, branch3x3dbl_2, branch3x3dbl_3, branch_pool], axis=-1, name='Inception_Block_C'+str(i))

    return out


def Reduction_Block_A(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block A (i)
    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2))

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out


def Reduction_Block_B(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block B (i)
    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2))

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (1, 7))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (7, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out


class SEInception:
    def __init__(self, length, width, num_channel, num_filters, ratio=4, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False, auxilliary_outputs=False):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Model
        # kernel_size: Kernel or Filter Size of the Input Convolutional Layer
        # num_channel: Number of Channels of the Input Predictor Signals
        # problem_type: Regression or Classification
        # output_nums: Number of Output Classes in Classification mode and output features in Regression mode
        # pooling: Choose either 'max' for MaxPooling or 'avg' for Averagepooling
        # dropout_rate: If turned on, some layers will be dropped out randomly based on the selected proportion
        # auxilliary_outputs: Two extra Auxullary outputs for the Inception models, acting like Deep Supervision
        self.length = length
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.ratio = ratio
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.auxilliary_outputs = auxilliary_outputs

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def SEInception_v1(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, self.num_filters, 7, strides=2)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv_2D_Block(x, self.num_filters, 1, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 3, 3)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Inceptionv1_Module(x, 64, 96, 128, 16, 32, 32, 1)  # Inception Block 1
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 128, 128, 192, 32, 96, 64, 2)  # Inception Block 2
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Inceptionv1_Module(x, 192, 96, 208, 16, 48, 64, 3)  # Inception Block 3
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 160, 112, 224, 24, 64, 64, 4)  # Inception Block 4
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 128, 128, 256, 24, 64, 64, 5)  # Inception Block 5
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 112, 144, 288, 32, 64, 64, 6)  # Inception Block 6
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 256, 160, 320, 32, 128, 128, 7)  # Inception Block 7
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 64, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Inceptionv1_Module(x, 256, 160, 320, 32, 128, 128, 8)  # Inception Block 8
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv1_Module(x, 384, 192, 384, 48, 128, 128, 9)  # Inception Block 9
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v1')

        return model

    def SEInception_v2(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem: 56 x 64
        x = tf.keras.layers.SeparableConv2D(self.num_filters, kernel_size=7, strides=(2, 2), depth_multiplier=1, padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv_2D_Block(x, self.num_filters * 2, 1, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 6, 3, padding='valid')
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Inceptionv2_Module(x, 64, 64, 64, 64, 96, 96, 32, 1)  # Inception Block 1: 28 x 192
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv2_Module(x, 64, 64, 96, 64, 96, 96, 64, 2)  # Inception Block 2: 28 x 256
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 128, 160, 64, 96, 96, 1)  # Reduction Block 1: 28 x 320

        x = Inceptionv2_Module(x, 224, 64, 96, 96, 128, 128, 128, 3)  # Inception Block 3: 14 x 576
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv2_Module(x, 192, 96, 128, 96, 128, 128, 128, 4)  # Inception Block 4: 14 x 576
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv2_Module(x, 160, 128, 160, 128, 160, 160, 96, 5)  # Inception Block 5: 14 x 576
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv2_Module(x, 96, 128, 192, 160, 192, 192, 96, 6)  # Inception Block 6: 14 x 576
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 192, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 128, 192, 192, 256, 256, 2)  # Reduction Block 2: 14 x 576

        x = Inceptionv2_Module(x, 352, 192, 320, 160, 224, 224, 128, 7)  # Inception Block 7: 7 x 1024
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inceptionv2_Module(x, 352, 192, 320, 192, 224, 224, 128, 8)  # Inception Block 8: 7 x 1024
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v2')

        return model

    def SEInception_v3(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, self.num_filters, 3, strides=2, padding='valid')
        x = Conv_2D_Block(x, self.num_filters, 3, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 2, 3)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Conv_2D_Block(x, self.num_filters * 2.5, 1, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 6, 3, padding='valid')
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # 3x Inception-A Blocks
        x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 32, 1)  # Inception-A Block 1: 35 x 256
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 64, 2)  # Inception-A Block 2: 35 x 256
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 64, 3)  # Inception-A Block 3: 35 x 256
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 64, 384, 64, 96, 96, 1)  # Reduction Block 1: 17 x 768

        # 4x Inception-B Blocks
        x = Inception_Module_B(x, 192, 128, 192, 128, 128, 192, 192, 1)  # Inception-B Block 1: 17 x 768
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_B(x, 192, 160, 192, 160, 160, 192, 192, 2)  # Inception-B Block 2: 17 x 768
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_B(x, 192, 160, 192, 160, 160, 192, 192, 3)  # Inception-B Block 3: 17 x 768
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_B(x, 192, 192, 192, 192, 192, 192, 192, 4)  # Inception-B Block 4: 17 x 768
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 192, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 192, 320, 192, 192, 192, 2)  # Reduction Block 2: 8 x 1280

        # 2x Inception-C Blocks: 8 x 2048
        x = Inception_Module_C(x, 320, 384, 384, 448, 384, 384, 192, 1)  # Inception-C Block 1: 8 x 2048
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = Inception_Module_C(x, 320, 384, 384, 448, 384, 384, 192, 2)  # Inception-C Block 2: 8 x 2048
        x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v3')

        return model

    def SEInception_v4(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, 32, 3, strides=2, padding='valid')
        x = Conv_2D_Block(x, 32, 3, padding='valid')
        x = Conv_2D_Block(x, 64, 3)

        branch1 = Conv_2D_Block(x, 96, 3, strides=2, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = Conv_2D_Block(x, 64, 1)
        branch1 = Conv_2D_Block(branch1, 96, 3, padding='valid')
        branch2 = Conv_2D_Block(x, 64, 1)
        branch2 = Conv_2D_Block(branch2, 64, 7)
        branch2 = Conv_2D_Block(branch2, 96, 3, padding='valid')
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = Conv_2D_Block(x, 192, 3, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        # 4x Inception-A Blocks - 35 x 256
        for i in range(4):
            x = Inception_Module_A(x, 96, 64, 96, 64, 96, 96, 96, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 96, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 64, 384, 192, 224, 256, 1)  # Reduction Block 1: 17 x 768

        # 7x Inception-B Blocks - 17 x 768
        for i in range(7):
            x = Inception_Module_B(x, 384, 192, 256, 192, 224, 256, 128, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 192, 192, 256, 320, 320, 2)  # Reduction Block 2: 8 x 1280

        # 3x Inception-C Blocks: 8 x 2048
        for i in range(3):
            x = Inception_Module_C(x, 256, 384, 512, 384, 512, 512, 256, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v4')
        if self.auxilliary_outputs:
            model = tf.keras.layers.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v4')

        return model



    
def making_training_and_testing_data(full_data,full_label):
    
    
    train_label=[]
    for i in range(len(full_label)):
        if full_label[i]==0:
            train_label.append([0,1,0])
        elif full_label[i]==1:
            train_label.append([1,0,0])
        elif full_label[i]==2:

            train_label.append([0,0,1])

    
    full_label=np.array(train_label)
    
    
    return full_data,full_label
    
def my_plots(history,my_model):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    my_path="training and validation accuracy curve of "+my_model+".png"
    plt.savefig(my_path)
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim([0, 1])

    #plt.ylim([-3, 3])
    plt.yticks(np.arange(0, 1.1, 0.25))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    my_path="training and validation loss curve of "+my_model+".png"
    plt.savefig(my_path)
    plt.show()

    
    
def SE_Block(inputs, num_filters, ratio):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    excitation = tf.keras.layers.Dense(units=num_filters/ratio)(squeeze)
    excitation = tf.keras.layers.Activation('relu')(excitation)
    excitation = tf.keras.layers.Dense(units=num_filters)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([1, 1, num_filters])(excitation)

    scale = inputs * excitation
    return scale

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Conv2D(squeeze, (1, 1), padding='valid')(x)
    x = Activation('relu')(x)

    left = Conv2D(expand, (1, 1), padding='valid')(x)
    left = Activation('relu')(left)

    right = Conv2D(expand, (3, 3), padding='same')(x)
    right = Activation('relu')(right)

    x = concatenate([left, right], axis=channel_axis)
    return x
from keras.utils import get_file
def SE_SQUEEZNET(inputs,ratio,num_of_class):
    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(inputs)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    
    
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2')(x)
    x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)
    x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    
    
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)    
    

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
    x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)  

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    #x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x=SE_Block(x,num_filters=int(np.shape(x)[-1]),ratio=ratio)
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_of_class, activation='softmax')(x)
    model = tf.keras.Model(inputs, [output])
    
    
    return model
    
if __name__ == '__main__':
    normal_dir = "" #give your normal cases data path here
    #vit_datasets/Dataset_ViT/ViT_dataset/Covid-19
    dir1 = os.path.join(normal_dir,"*.npy")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    dir = os.path.join(normal_dir,"*.jpg")
    normal_files = glob.glob(dir)
    normal_1 = glob.glob(dir1)
    normal_2 = glob.glob(dir2)
    normal_files.extend(normal_1)
    normal_files.extend(normal_2)

    normal_dir = ""  #give your covid 19 cases data path here
    dir1 = os.path.join(normal_dir,"*.npy")
    dir = os.path.join(normal_dir,"*.jpg")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    covid_files = glob.glob(dir)
    covid_files2 = glob.glob(dir2)
    covid_files1 = glob.glob(dir1)
    covid_files.extend(covid_files2)
    covid_files.extend(covid_files1)

    normal_dir = "" #give your pneumonia cases data path here
    dir1 = os.path.join(normal_dir,"*.npy")
    dir2 = os.path.join(normal_dir,"*.jpeg")
    dir = os.path.join(normal_dir,"*.jpg")
    pneumonia_files = glob.glob(dir)
    pneumonia_1 = glob.glob(dir1)
    pneumonia_2 = glob.glob(dir2)
    pneumonia_files.extend(pneumonia_1)
    pneumonia_files.extend(pneumonia_2)

    normal_files.sort()
    covid_files.sort()
    pneumonia_files.sort()
    normal_files=shuffle(normal_files,random_state=10)
    covid_files=shuffle(covid_files,random_state=10)
    pneumonia_files=shuffle(pneumonia_files,random_state=10)
    print("pneumonia_files:",len(pneumonia_files))
    print("covid_files:",len(covid_files))
    print("normal_files:",len(normal_files))
    x=(len(normal_files)+len(covid_files)+len(pneumonia_files))*0.20
    y=(len(normal_files)+len(covid_files)+len(pneumonia_files))*0.10
    x=int(x//3)
    y=int(y//3)
    print(x)
    print(y)
    
    for_normal=x-200
    for_covid=x+150
    for_pneumonia=x+50
    test_normal_files=normal_files[:for_normal]
    test_covid_files=covid_files[:for_covid]
    test_pneumonia_files=pneumonia_files[:for_pneumonia]
    
    val_normal_files=normal_files[for_normal:for_normal+y]
    val_covid_files=covid_files[for_covid:for_covid+y]
    val_pneumonia_files=pneumonia_files[for_pneumonia:for_pneumonia+y]
    
    train_normal_files=normal_files[for_normal+y:]
    train_covid_files=covid_files[for_covid+y:]
    train_pneumonia_files=pneumonia_files[for_pneumonia+y:]
    print("test normal:",len(test_normal_files))
    print("test covid:",len(test_covid_files))
    print("test pneumonia:",len(test_pneumonia_files))
    print("val normal:",len(val_normal_files))
    print("val covid:",len(val_covid_files))
    print("val pneumonia:",len(val_pneumonia_files))
    print("train normal:",len(train_normal_files))
    print("train covid:",len(train_covid_files))
    print("train pneumonia:",len(train_pneumonia_files))
    
    
    
    train_aug_normal,train_aug_covid,train_aug_pneumonia=data_augmentation(train_normal_files,train_covid_files,train_pneumonia_files)
    test_aug_normal,test_aug_covid,test_aug_pneumonia=no_data_augmentation(test_normal_files,test_covid_files,test_pneumonia_files)
    val_aug_normal,val_aug_covid,val_aug_pneumonia=no_data_augmentation(val_normal_files,val_covid_files,val_pneumonia_files)
    
    train_full_data,train_full_label=making_full_data(train_aug_normal,train_aug_covid,train_aug_pneumonia)  #getting my full data
    test_full_data,test_full_label=making_full_data(test_aug_normal,test_aug_covid,test_aug_pneumonia)
    val_full_data,val_full_label=making_full_data(val_aug_normal,val_aug_covid,val_aug_pneumonia)
    
    train_full_data,train_full_label= making_training_and_testing_data(train_full_data,train_full_label) #dividing full_data into train and test data
    test_full_data,test_full_label=making_training_and_testing_data(test_full_data,test_full_label)
    val_full_data,val_full_label=making_training_and_testing_data(val_full_data,val_full_label)
    
    if model_name==1:  #IT WILL RUN DENSENET 201 MODEL
        Model = Sequential()

        pretrained_model= tf.keras.applications.DenseNet201(include_top=False,   #insitializing our modl
                           input_shape=(224,224,3),
                           classes=3,
                           weights='imagenet')
        for layer in pretrained_model.layers:
                layer.trainable=False

        Model.add(pretrained_model)
        Model.add(Flatten())
        Model.add(Dense(512, activation='relu'))
        Model.add(Dense(3, activation='softmax'))
        Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False), metrics=['accuracy'])
    #inception_pred.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0002), loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False) , metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', 
            patience=35, 

            min_delta=0.001, 
            mode='min')
        print("X_train - 3* multiplying")
        train_data=np.stack((train_full_data,)*3,axis=-1)
        print("X_test - 3* multiplying")
        test_data=np.stack((val_full_data,)*3,axis=-1)
        history=Model.fit(train_data,train_full_label,epochs=100,validation_data=(test_data, val_full_label),callbacks=[early_stopping],batch_size=32)
        my_plots(history,"DENSENET_201_with_segmentation")
        filename = 'densent_201_model_with_segmentation.sav'
        np.save('densent_201_model_with_segmentation.npy',history.history)
        pickle.dump(Model, open(filename, 'wb'))
    elif model_name==2: #IT WILL RUN SE INCEPTION V3 MODEL
      
        length = 224  # Length of each Image
        width = 224  # Width of each Image
        model_name = 'SEInceptionV3'  # DenseNet Models
        model_width = 64 # Width of the Initial Layer, subsequent layers start from here
        num_channel = 1  # Number of Input Channels in the Model
        problem_type = 'Classification' # Classification or Regression
        output_nums = 3  # Number of Class for Classification Problems, always '1' for Regression Problems
        reduction_ratio = 16
        #
        Model = SEInception(length, width, num_channel, model_width, ratio=reduction_ratio, problem_type=problem_type, output_nums=output_nums,
                          pooling='avg', dropout_rate=0.2, auxilliary_outputs=False).SEInception_v3()
        Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False), metrics=['accuracy'])
        #inception_pred.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0002), loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False) , metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', 
            patience=35, 

            min_delta=0.001, 
            mode='min')
        history=Model.fit(train_full_data,train_full_label,epochs=100,validation_data=(val_full_data, val_full_label),callbacks=[early_stopping],batch_size=32) #FITTING THE MODEL

        my_plots(history,"SE_INCEPTION_V3_with_segmentation") #PLOTTING THE ACCURACY AND LOSS CURVE

        filename = 'SE_INCEPTION_V3_MODEL_with_segmentation.sav'
        np.save('SE_INCEPTION_V3_MODEL_with_segmentation.npy',history.history)
        pickle.dump(Model, open(filename, 'wb')) #SAVING THE MODEL
    
    elif model_name==3: #IT WILL RUN SE SQUEEZENET MODEL
        input_ = Input(shape =(224,224,1))
        sq1x1 = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu = "relu_"
        ratio=8
        num_of_classes=3
        Model = SE_SQUEEZNET(inputs=input_,ratio=ratio,num_of_class=num_of_classes)


        early_stopping = EarlyStopping(monitor='val_loss', 
            patience=40, 
            min_delta=0.001, 
            mode='min')
        print("Multipluing 3 times")
        #train_data=np.stack((train_full_data,)*3,axis=-1)
        train_data=train_full_data
        test_data=val_full_data
            #val_data=np.stack((val_data,)*3,axis=-1)
        #test_data=np.stack((val_full_data,)*3,axis=-1)
        print("multiplying done")
        Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False), metrics=['accuracy'])
            #fitting the model
        history=Model.fit(train_data,train_full_label,epochs=100,validation_data=(test_data, val_full_label),callbacks=[early_stopping],batch_size=32)

        my_plots(history,"se_squeeznet_with_segmentation") #plotting the accuracy and loss curve

        filename = 'se_squeeznet_MODEL_with_segmentation.sav' #file name with which model is saved
        np.save('se_squeeznet_MODEL_with_segmentation.npy',history.history)
        pickle.dump(Model, open(filename, 'wb')) #model is saved
    else:
        print("Check your input model - only 1,2,3  are allowed")
