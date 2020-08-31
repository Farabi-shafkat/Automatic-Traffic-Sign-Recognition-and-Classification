#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#https://docs.python-guide.org/dev/virtualenvs/
#venv\Scripts\activate

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


class TrafficSignNet:
    @staticmethod
    def build(width,height,depth,classes):
        #build and initialize the model
        model=Sequential()
        inputshape=(height,width,depth)
        chanDim=-1
        #CONV => RELU => BN => POOL
        model.add(Conv2D(8,(5,5),padding="same",input_shape=inputshape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        #(CONV => RELU => CONV => RELU) => POOL

        model.add(Conv2D(16,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(32,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #Two sets of fully connected layers and a softmax classifier

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Flatten()) #is this even necessary?
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


