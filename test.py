import cv2
import pickle
import os 
from keras.models import load_model
import numpy as np
from keras.utils import np_utils

path = '/home/dlagroup7/Assignment2_Group3/data/input/Test_Data/'
# exit()
model = load_model('31i.h5')

def test_data():
	for file in os.listdir(path):
		print file
		data = pickle.load(open(path + '/' + file, "rb" ))
		test, count = [], 0
		for img in data:
			r_img = cv2.resize(img, (224,224))
			test.append(r_img)
			count = count+1
		print 'total : ',count
		test = np.asarray(test)
		test = test.reshape(test.shape[0], 224, 224,3).astype('float32')
		scr = model.predict(test)
		f_name = file+'.txt'
		with open(f_name, 'w+') as fp:
			for s in scr:
				fp.write(str(s[1])+'\n')

test_data()

'''
test = []
print 'main'
model = load_model('all_res20.h5')
print model.summary()

img_names = ['11_3_1.tif','11_6_2.tif','11_4_2.tif','11_5_1.tif','11_5_2.tif']
for im in img_names:
	img = cv2.imread(im)		
	img = cv2.resize(img, (224,224))
	img = np.asarray(img)
	# img = img.reshape(1, 224, 224,3).astype('float32')
	test.append(img)
test = np.asarray(test)
test = test.reshape(test.shape[0], 224, 224,3).astype('float32')
scr = model.predict(test, verbose=1)
print scr[]
# ind = np.argmax(scr, axis=1)
# print ind
	# print argmax(scr)
	# test.append(resize_img)

	# resize_img = resize_img.reshape(1, 224, 224,3).astype('float32')
# test = np.asarray(test)
# test = test.reshape(test.shape[0], 224, 224,3).astype('float32')
# scr = model.evaluate(test,test_y,verbose=1)
# print scr
# i = 0
# for files_ in all_files:
# 	prob = model.predict(files_, batch_size = 16)
# 	name = names[i][:4] + '.txt'
# 	with open(name, 'r+') as fp:
# 		for p in prob:
# 			fp.write(p[0] + '\n')

'''
