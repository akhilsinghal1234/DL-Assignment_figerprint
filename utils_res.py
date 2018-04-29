import cv2
import os
import pickle

live,spoof = [],[]
path = '../../training_data'

def progress(dataset):
	print('Reading -',dataset)

all_ext = [".png", ".tif", ".tiff", ".jpeg", ".bmp"]

def get_images(img_path, k):
	for imgname in os.listdir(img_path):
		if imgname.endswith(tuple(all_ext)):
			img = cv2.imread(img_path + "/" + imgname)		
			if img is not None:
				resize_img = cv2.resize(img, (224,224))
				if k == 1:
					live.append([resize_img, img_path + "/" + imgname])
				else:
					spoof.append([resize_img, img_path + "/" + imgname])
def get_train():
	for dataset in os.listdir(path):
		if dataset == "LivDet2009":
			progress(dataset)
			for sensor in os.listdir(path+"/" + dataset):
				if sensor == "Biometrika":
					for folder in os.listdir(path+"/" + dataset + "/" + sensor):
						if folder == "Alive":
							img_path = path+"/" + dataset + "/" + sensor+ "/" + folder
							get_images(img_path,1)

						else:
							img_path = path+"/" + dataset + "/" + sensor+ "/" + folder
							get_images(img_path,0)
				else:
					for folder in os.listdir(path+"/" + dataset + "/" + sensor):
						if folder == "Alive":
							folder_path = path+"/" + dataset + "/" + sensor + "/" + folder
							for s_folder in os.listdir(folder_path):
								img_path = folder_path + "/" + s_folder
								get_images(img_path,1)
						else:
							folder_path = path+"/" + dataset + "/" + sensor + "/" + folder
							for s_folder in os.listdir(folder_path):
								img_path = folder_path + "/" + s_folder
								get_images(img_path,0)

		if dataset == "LivDet2011":
			progress(dataset)
			for sensor in os.listdir(path+"/" + dataset):
				for folder in os.listdir(path+"/" + dataset + "/" + sensor):
					if folder == "Live":
						img_path = path+"/" + dataset + "/" + sensor+ "/" + folder
						get_images(img_path,1)

					else:
						for spooftype in os.listdir(path+"/" + dataset + "/" + sensor+ "/" + folder):
							img_path = path+"/" + dataset + "/" + sensor+ "/" + folder + "/" + spooftype
							get_images(img_path,0)

		elif dataset == "LivDet2013":
			progress(dataset)
			for sensor in os.listdir(path+"/" + dataset):
				for folder in os.listdir(path+"/" + dataset + "/" + sensor):
					if folder == "Live":
						img_path = path+"/" + dataset + "/" + sensor+ "/" + folder
						get_images(img_path,1)

					else:
						for spooftype in os.listdir(path+"/" + dataset + "/" + sensor+ "/" + folder):
							img_path = path+"/" + dataset + "/" + sensor+ "/" + folder + "/" + spooftype
							get_images(img_path,0)

		elif dataset == "LivDet2015":
			progress(dataset)
			for sensor in os.listdir(path+"/" + dataset):
				for folder in os.listdir(path+"/" + dataset + "/" + sensor):
					if folder == "Live":
						img_path = path+"/" + dataset + "/" + sensor+ "/" + folder
						get_images(img_path,1)

					else:
						for spooftype in os.listdir(path+"/" + dataset + "/" + sensor+ "/" + folder):
							img_path = path+"/" + dataset + "/" + sensor+ "/" + folder + "/" + spooftype
							get_images(img_path,0)

	return live,spoof
