# Automatic Traffic Sign Recognition and Classification
This is code I wrote for the following project: https://drive.google.com/file/d/1XJr7LQCJ3y_t8RCASXomNDypCScAVXUh/view?usp=sharing

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
  python predict.py --model output/trafficsignnet.model --images gtsrb-german-traffic-sign/Test --examples examples
If you want to test the model on custom images, set the --images argument to the path of the images. After testing, the scipt will save randomly chosen examples in the --examples path
