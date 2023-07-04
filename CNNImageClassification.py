import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
data_dir='data'
image_exts=['jpeg','jpg','bmp','png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path=os.path.join(data_dir,image_class,image)
        try:
            img=cv2.imread(image_path)
            tip=imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("Issue with image {}".format(image_path))
            os.remove(image_path)
data=tf.keras.utils.image_dataset_from_directory('data')
data=data.shuffle(buffer_size=1000)
data_iterator=data.as_numpy_iterator()
batch=data_iterator.next()
fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx]) #0 bagels 1 puppies
data=data.map(lambda x,y:(x/255,y)) # makes all values between 0 and 1, x represents image, y represents labels
data_length=len(data)
train_size=int(data_length*.7)
val_size=int(data_length*.2) #model evaluates as it trains
test_size=int(data_length*.1) #check each one and add +1 or something if appropriate

train=data.take(train_size)
validation=data.skip(train_size).take(val_size)
test=data.skip(train_size+val_size).take(test_size)

model=Sequential()
model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])

logdir='logs'
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir) #history of training kind of
hist=model.fit(train,epochs=20,validation_data=validation,callbacks=[tensorboard_callback])

loss_fig=plt.figure()
plt.plot(hist.history['loss'],color='teal',label='loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
loss_fig.suptitle('Loss',fontsize=20)
plt.legend(loc='upper left')
plt.show()

accuracy_fig=plt.figure()
plt.plot(hist.history['accuracy'],color='teal',label='accuracy')
plt.plot(hist.history['val_loss'],color='orange',label='val_accuracy')
loss_fig.suptitle('Accuracy',fontsize=20)
plt.legend(loc='upper left')
plt.show()

model.save("C:/Users/Jensen/TrainedDogsVBagelModel/TrainedDogsVBagelModel.h5")


