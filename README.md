# traffic-sign-recognition-tutorial-code
This is the code i wrote while following a tutorial made by pyimagsearch that uses keras to train a deep convolutional neural network to recognise traffic signs
The full tutorial can be found here : https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/

Traffic sign classification is the process of automatically recognizing traffic signs along the road, including speed limit signs, yield signs, merge signs, etc. Being able to automatically recognize traffic signs enables us to build “smarter cars”.

Self-driving cars need traffic sign recognition in order to properly parse and understand the roadway. Similarly, “driver alert” systems inside cars need to understand the roadway around them to help aid and protect drivers.

Traffic sign recognition is just one of the problems that computer vision and deep learning can solve.

The dataset used to train and test this deep neural network is  German Traffic Sign Recognition Benchmark (GTSRB).It can be downloaded from https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
Create a new folder in teh project root called data and place the extracted dataset there.


## Configuring-your-development-environment


run these commands install all the dependencies:

  pip install opencv-contrib-python
  
  pip install numpy
  
  pip install scikit-learn
  
  pip install scikit-image
  
  pip install imutils
  
  pip install matplotlib
  
  pip install tensorflow==2.0.0 # or tensorflow-gpu
  

## Tarining the model:
Open a command promt and run the command below:
  python train.py --dataset gtsrb-german-traffic-sign --model output/trafficsignnet.model --plot output/plot.png
 
 
 
## Predictions:
Open up a terminal and execute the following command:
  ython predict.py --model output/trafficsignnet.model --images gtsrb-german-traffic-sign/Test --examples examples
