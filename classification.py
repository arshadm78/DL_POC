import pandas as pd
import numpy as np
import seaborn as sns
import warnings, os, pickle, math, pdb

import matplotlib.pyplot as plt
from vis.utils import utils
from vis.input_modifiers import Jitter

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks  import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import image
from keras.models import load_model
from keras import backend as K, activations


from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocessor_input
from keras.applications.inception_v3 import decode_predictions as inceptionv3_decode_predictions

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocessor_input
from keras.applications.vgg16 import decode_predictions as vgg16_decode_predictions

from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocessor_input
from keras.applications.vgg19 import decode_predictions as vgg19_decode_predictions

from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocessor_input
from keras.applications.resnet50 import decode_predictions as resnet50_decode_predictions

global model
global modelvgg16
global modelvgg19
global modelinceptionv3
global modelresnet50
global input_width
global input_height
global decode_predictions
global preprocess_input

plt.style.use('ggplot')
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

model = None
modelvgg16 = None
modelvgg19 = None
modelinceptionv3 = None
modelresnet50 = None

def switchvgg16():
    global model
    global modelvgg16
    global input_width
    global input_height
    global decode_predictions
    global preprocess_input
    
    input_width = 224
    input_height = 224
    model = modelvgg16
    preprocess_input = vgg16_preprocessor_input
    decode_predictions = vgg16_decode_predictions
     
def switchvgg19():
    global model
    global modelvgg19
    global input_width
    global input_height
    input_width = 224
    input_height = 224
    model = modelvgg19
    preprocess_input = vgg19_preprocessor_input
    decode_predictions = vgg19_decode_predictions
     
    
def switchinceptionv3():
    global model
    global modelinceptionv3
    global input_width
    global input_height
    global decode_predictions
    global preprocess_input
    
    input_width = 299
    input_height = 299
    model = modelinceptionv3
    preprocess_input = inceptionv3_preprocessor_input
    decode_predictions = inceptionv3_decode_predictions
    
    
def switchresnet50():
    global model
    global modelresnet50
    global input_width
    global input_height
    global decode_predictions
    global preprocess_input
    
    input_width = 224
    input_height = 224
    model = modelresnet50
    preprocess_input = resnet50_preprocessor_input
    decode_predictions = resnet50_decode_predictions
    
    
def loadinceptionv3():
    global modelinceptionv3
    
    print "Loading Inception ... "
    if modelinceptionv3 is None:
        modelinceptionv3 = InceptionV3(weights='imagenet')
    print "Model loaded"
   
    switchinceptionv3()
    
def loadresnet50():
    global modelresnet50
    
    print "Loading Resnet50 ... "
    if modelresnet50 is None:
        modelresnet50 = ResNet50(weights='imagenet')
    print "Model loaded"
    
    switchresnet50()
    
def loadvgg16():
    global modelvgg16

    print "Loading VGG16 ... "
    if modelvgg16 is None:
        modelvgg16 =  VGG16(weights='imagenet')
    print "Model loaded"
    switchvgg16()

def loadvgg19():
    global modelvgg19
    
    print "Loading VGG19 ... "
    if modelvgg19 is None:
        modelvgg19 =  VGG19(weights='imagenet')
    print "Model loaded"
    switchvgg19()
    
    
def doinference(path):
    global model
    img_path = path
    img = image.load_img(img_path, target_size=(input_width, input_height))
    x = image.img_to_array(img)
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    #x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:',  decode_predictions(preds))
    
file='/home/arshad/projects/Paper/zebra.jpg'
loadvgg16()
doinference(file)
loadvgg19()
doinference(file)
loadinceptionv3()
doinference(file)
loadresnet50()
doinference(file)

