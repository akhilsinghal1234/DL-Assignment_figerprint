import numpy
from keras.applications.resnet50 import ResNet50
from operator import itemgetter
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import optimizers
from sklearn.metrics import confusion_matrix
import copy

K.set_image_dim_ordering('tf')

from utils_res import get_train

datasets = ['LivDet2009','LivDet2011','LivDet2013','LivDet2015']
sensor = [None]*4
sensor[0] = ['Biometrika', 'CrossMatch', 'Identix']
sensor[1] = ['Biometrika', 'Digital', 'Italdata', 'Sagem']
sensor[2] = ['Biometrika', 'CrossMatch', 'Italdata', 'Swipe']
sensor[3] = ['CrossMatch', 'Digital_Persona', 'GreenBit','Hi_Scan', 'Time_Series']

results = []
for i in range(len(datasets)):
	results.append([0]*len(sensor[i]))
# print(results)
original = copy.deepcopy(results)

live,spoof = get_train()
# print 'in main'
l = len(live)
s = len(spoof)
live = numpy.asarray(live)
spoof = numpy.asarray(spoof)
live_y, spoof_y = [1 for x in range(l)], [0 for x in range(s)]
data1 = numpy.concatenate([live,spoof])
data1_y = numpy.concatenate([live_y,spoof_y])
data,data_y = shuffle(data1,data1_y, random_state=42)

train1, test1, train_y, test_y = train_test_split(data, data_y, test_size=0.10, random_state=42)

train,test = [],[]
for tr in train1:
	train.append(tr[0])
file_path = []
for te in test1:
	test.append(te[0])
	file_path.append(te[1])

train = numpy.asarray(train)
test = numpy.asarray(test)
train = train.reshape(train.shape[0], 224, 224,3).astype('float32')
test = test.reshape(test.shape[0], 224, 224,3).astype('float32')	
# train = train / 255
# test = test / 255
train_y = np_utils.to_categorical(train_y)
test_y1 = copy.deepcopy(test_y)
test_y = np_utils.to_categorical(test_y)

num_classes = 2

model = ResNet50()
# print model.layers[:14]
length = len(model.layers)

for i in range(length-7):					# till last conv block
    model.layers[i].trainable = False

fc2 = model.get_layer('flatten_1').output
fc3 = Dense(1024, activation='relu', name='2')(fc2)
D = Dropout(0.1)(fc3)
fc4 = Dense(1024,activation='relu', name='3')(D)
# D1 = Dropout(0.2)(fc4)
predictions = Dense(2,activation='softmax', name='predict')(fc4)
model1 = Model(model.input, predictions)

# datagen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    # rotation_range=5,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=False)
#     )

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 	
model1.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# datagen.fit(train)
# epochs = 40
# for epoch in range(epochs)
# model1.fit(train, train_y, epochs=60, batch_size=100, verbose=1)
# model1.fit_generator(datagen.flow(train,train_y, batch_size=64,save_to_dir='augs'), steps_per_epoch=len(train_y) / 64, epochs=epochs,verbose=1,validation_data=(test, test_y))
epochs=50
# t = 1
for e in range(epochs):
	model1.fit(train, train_y, epochs=1, batch_size=100, verbose=1)
	scores = model1.evaluate(test, test_y, verbose=1)
	print e,'-',scores[1]*100
	if scores[1]*100 > 98.90:
		model1.save(str(e)+'.h5')
		print 'saved',str(e)+'.h5','for',scores[1]*100
# Final evaluation of the model

scores = model1.evaluate(test, test_y, verbose=1)
print "Resnet Error: %.2f%%" % (100-scores[1]*100)
# print "Resnet Accuracy: %.2f%%" % (scores[1]*100)
# model1.save('res60.h5')

