import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure

#cascade.xml contains features needed to find traffic signs
cascade = cv2.CascadeClassifier('cascade.xml')

img = cv2.imread('input.jpg')
full_image = np.array(img)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # no need to convert to gray

#loading conv model
print("[INFO] loading model....")
#set model path here
model = load_model('/content/traffic-sign-recognition-tutorial-code/output/trafficsignnet.model')
labelnames = open("signnames.csv").read().strip().split("\n")[1:]
labelnames = [l.split(",")[1] for l in labelnames]

#detection
tic = time.time()
boxes = cascade.detectMultiScale(img, scaleFactor = 1.01, minNeighbors = 7, minSize= (24,24), maxSize=(128,128)) 
tac = time.time()
print('detection time : ', (tac - tic)*1000 ,'seconds')

#drawing boundary boxes on input image
for (x,y,w,h) in boxes:
    print(x,y,w,h)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    #model.predict
    cropped_image = full_image[x:x+w, y:y+h, : ]
    cropped_image = transform.resize(cropped_image,(32,32))
    cropped_image = exposure.equalize_adapthist(cropped_image, clip_limit=0.1)
    cropped_image = cropped_image.astype("float")/255.0
    cropped_image = np.expand_dims(cropped_image, axis=0)
    preds = model.predict(cropped_image)
    j = preds.argmax(axis=1)[0]
    label = labelnames[j]
    print(" j:",j," max_pred:",preds.max(), " label:",label)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.45, (0, 255, 255), 2)
    
#saving output file
cv2.imwrite('output.jpg', img)
