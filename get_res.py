import numpy
from keras.utils import np_utils
from keras import backend as K
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras import optimizers
from sklearn.metrics import confusion_matrix
import copy
from keras.models import load_model

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
results_lf = copy.deepcopy(results)
results_l = copy.deepcopy(results)
results_f = copy.deepcopy(results)

live,spoof = get_train()

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
file_path = []
for te in test1:
	test.append(te[0])
	file_path.append(te[1])

test = numpy.asarray(test)

test = test.reshape(test.shape[0], 224, 224,3).astype('float32')	

test_y1 = copy.deepcopy(test_y)
test_y = np_utils.to_categorical(test_y)

num_classes = 2

model = load_model('31i.h5')

scores = model.evaluate(test, test_y, verbose=1)
print "Resnet Error: %.2f%%" % (100-scores[1]*100)
print "Resnet Accuracy: %.2f%%" % (scores[1]*100)

scores = model.predict(test)
scr_class = numpy.argmax(scores, axis=1)

for i in range(len(test_y1)):
	if test_y1[i] == 0:
		for j in range(len(datasets)):
			if datasets[j] in file_path[i]:
				for k in range(len(sensor[j])):
					if sensor[j][k] in file_path[i]:
						results_f[j][k] = results_f[j][k] + 1
	if test_y1[i] == 1:
		for j in range(len(datasets)):
			if datasets[j] in file_path[i]:
				for k in range(len(sensor[j])):
					if sensor[j][k] in file_path[i]:
						results_l[j][k] = results_l[j][k] + 1

misclassified_files0,misclassified_files1 = [],[]
for i in range(len(test_y1)):
	if scr_class[i] == 0 and test_y1[i] == 1:
		misclassified_files0.append(file_path[i])
		for j in range(len(datasets)):
			if datasets[j] in file_path[i]:
				for k in range(len(sensor[j])):
					if sensor[j][k] in file_path[i]:
						results_lf[j][k] = results_lf[j][k] + 1

	if scr_class[i] == 1 and test_y1[i] == 0:
		misclassified_files1.append(file_path[i])
		for j in range(len(datasets)):
			if datasets[j] in file_path[i]:
				for k in range(len(sensor[j])):
					if sensor[j][k] in file_path[i]:
						results[j][k] = results[j][k] + 1

	for j in range(len(datasets)):
		if datasets[j] in file_path[i]:
			for k in range(len(sensor[j])):
				if sensor[j][k] in file_path[i]:
					original[j][k] = original[j][k] + 1

print 'fake but live:', results
print 'live but fake:', results_lf
print 'all_live:', results_l
print 'all_fake:', results_f
print 'all:', original

with open('miss_images_fake.txt','w+') as fp:					# live but predicted fake
	for p in misclassified_files0:
		fp.write(p+'\n')

with open('miss_images_live.txt','w+') as fp:					# fake but predicted live
	for k in misclassified_files1:
		fp.write(k+'\n')
