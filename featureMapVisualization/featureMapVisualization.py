import numpy as np
import cv2
import time
from skimage import io 
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
from numpy import expand_dims
import argparse


### HOW TO RUN
# python featureMapVisualization.py --model MODEL_PATH --image INPUT_IMAGE_PATH

### WHAT IT DOES
# it will run the model with given image and plot features maps from certain layers and also save them as images


ap = argparse.ArgumentParser()
ap.add_argument("-m","--model", required=True, help="path to the model")
ap.add_argument("-i","--image", required=True, help="path to the image")
args = vars(ap.parse_args())


image = io.imread(args["image"])    #image = io.imread('stop.png')
plt.imshow(image)
plt.title('input image')
plt.show()
image = transform.resize(image,(32,32))
image = exposure.equalize_adapthist(image, clip_limit=0.1)
image = image.astype("float")/255.0
image = np.expand_dims(image, axis=0)



model1 = load_model(args["model"])    #model1 = load_model('bestModel.h5')
indexes = [6,9,13,16]  
# index of the layers at the end of each conv block  (Conv>Activation>BN) of trafficSignNetModel
num_features = [4,4,8,8]
outputs = [model1.layers[i].output for i in indexes]
model2 = Model(inputs=model1.inputs, outputs=outputs)



feature_maps = model2.predict(image)



for ind,fmap in enumerate(feature_maps):
    ix = 1
    figtitle = 'output_of_layer_'+ str(indexes[ind],) + '_' + model1.layers[indexes[ind]].name  
    print(figtitle)
    for _ in range(4):
        for _ in range(num_features[ind]):
            ax = plt.subplot(4,num_features[ind],ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(fmap[0,:,:,ix-1], cmap = 'gray')
            ix += 1
    plt.savefig(figtitle + ".jpg")
    plt.show()
    