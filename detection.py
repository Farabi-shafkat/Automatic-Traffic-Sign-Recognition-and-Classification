from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io 
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import random
import imutils

# ap = argparse.ArgumentParser()
# ap.add_argument("-m","--model",required=True,help="path to the model")
# ap.add_argument("-i","--images",required=True,help="path to the images")
# ap.add_argument("-e","--examples",required=True,help="path to output example images")
# args = vars(ap.parse_args())


print("[INFO] loading model....")
model = load_model(args["model"])
labelnames = open("signnames.csv").read().strip().split("\n")[1:]
labelnames = [l.split(",")[1] for l in labelnames]

# print("[INFO] predicting...")
# imagepath = list(paths.list_images(args["images"])) 
# #print(imagepath)
# random.shuffle(imagepath)
# imagePaths = imagepath[:25]
#print(imagePaths)
# for (i,path) in enumerate(imagePaths):
#     print(i)
path = ""
full_image = io.imread(path)
w,h,c = full_image.shape()
patch_size = 32
stride = patch_size
output_image = full_image


for i in range(0,w - patch_size,stride):
    for j in range(0, h - patch_size, stride):
        image = full_image[i:i+patch_size, j:j+patch_size, : ]
        #image = transform.resize(image,(32,32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)

        image = image.astype("float")/255.0
        image = np.expand_dims(image, axis=0)

        preds = model.predict(image)
        if(preds.max()<0.90):
            break
        j = preds.argmax(axis=1)[0]
        label = labelnames[j]
        print("i:",i," j:",j," max_pred:",preds.max(), " label:",label)
        #image = cv2.imread(path)
        #image = imutils.resize(image, width=128)
        #cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
        #    0.45, (0, 0, 255), 2)
        #p = os.path.sep.join([args["examples"], "{}.png".format(i)])
        #cv2.imwrite(p, image)
        output_image = cv2.rectangle(output_image, (i,j), (i+32, j+32), (255,0,0), 2) 

cv2.imwrite("output.png", output_image)
print("[info] exiting")