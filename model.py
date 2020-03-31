# importing libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
import cv2
import os 
from random import shuffle
from tqdm import tqdm
import numpy as np 
from sklearn.model_selection import train_test_split

size_img = 224

img_width, img_height = 224, 224

path_train_data_neg = os.path.join(os.getcwd(),'all_neg')
path_train_data_pos = os.path.join(os.getcwd(),'all_pos')

print(os.listdir(path_train_data_neg))
path_neg = os.path.join(os.getcwd(),path_train_data_neg)
path_pos = os.path.join(os.getcwd(),path_train_data_pos)
# exit(0)

def prepare_data():
	x = [] # images as arrays
	y = [] # labels

	for image in tqdm(os.listdir(path_neg)):
		try:
			x.append(cv2.resize(cv2.imread(os.path.join(path_neg,image)), (size_img,size_img), interpolation=cv2.INTER_CUBIC))
			y.append(1)
		except Exception as e:
			print(str(e))
		
	for image in tqdm(os.listdir(path_pos)):
		try:
			x.append(cv2.resize(cv2.imread(os.path.join(path_pos,image)), (size_img,size_img), interpolation=cv2.INTER_CUBIC))
			y.append(0)
		except Exception as e:
			raise e
		

	return x,y

X, Y = prepare_data()
print(K.image_data_format())

X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=13)

nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 16


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=False)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=False)


train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)

history = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')


# ypred = model.predict(X_val)

import matplotlib.pyplot as plt 


fig = plt.figure()

def decode(val):
	if val == 1:
		return 'Negative'
	return 'Positive'

for num,data in enumerate(X_val[:9]):

	img_data = data.reshape(1,size_img,size_img,3)
	img_pred = decode(model.predict(img_data))
	img_true = decode(Y_val[num])

	yy = fig.add_subplot(3,3,num+1)

	str_label = 'True : '+ img_true +', Predicted : '+ img_pred

	yy.imshow(data,cmap='viridis')
	plt.title(str_label)
	yy.axes.get_xaxis().set_visible(False)
	yy.axes.get_yaxis().set_visible(False)


plt.show()


