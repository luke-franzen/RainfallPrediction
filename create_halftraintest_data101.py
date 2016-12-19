import h5py
import numpy as np
import scipy
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

#location of hdf5 file from 'createhdf5_101all.py' script
hdf = "/Users/lfranzen/bda/hdf5/data_101_all.hdf5"

#num frames to use to create model (how far back in time to use as model input)
num_train_frames = 12

#how far into the future to predict (keep this at 12 == 1 hour)
num_pred_frames = 12

#width of region being used to predict, must be odd number
box_width = 101

#index of label in flattened matrix
label_index = math.floor(box_width*box_width/2)

def main():

	train_data = np.zeros(shape=(280628, 30600), dtype=np.int8)
	train_index = 0
	train_labels = []
	train_label_dict = {}

	test_data = np.zeros(shape=(70341, 30600), dtype=np.int8)
	test_index = 0
	test_labels = []
	test_label_dict = {}

	rain_count = 0
	no_rain = 0
	train_files = 0
	total_files = 0

	with h5py.File(hdf, 'r') as hf:
		years = list(hf.keys())
		for year in years:
			months = list(hf[year].keys())
			for month in months:
				data = hf[year].get(month)
				keys = list(data.keys())
				num_files = len(keys)
				total_files += num_files

				#flatten matrices, create row vector for number of frames in past
				frame_num=0
				while (frame_num < len(keys)): 
					frame_data = np.array(data[keys[frame_num]])
					frame_data[frame_data < 0] = 0
					flat_data = frame_data.flatten().astype(np.int8)
					label = flat_data[label_index]
					if label != 0:
						rain_count += 1
					else:
						no_rain += 1
					train_label_dict[frame_num] = label

					#check boundaries for when to stop for train data and train labels
					if (frame_num+num_train_frames < num_files-num_pred_frames):
						j = 2
						final_data = np.delete(flat_data, label_index)
						tmp_arr = []
						tmp_arr.append(final_data)
						while (j < num_train_frames):
							frame_data = np.array(data[keys[j+frame_num]])
							frame_data[frame_data < 0] = 0
							flat_data = frame_data.flatten().astype(np.int8)
							final_data = np.delete(flat_data, label_index)
							tmp_arr.append(final_data)
							j+=2
						row = np.hstack(tmp_arr)
						start_ind = random.randint(0,1)
						row = row[start_ind::2]

						#if year is 2009-2012, append to train data
						#otherwise append to test data
						if year != '2013':
							train_data[train_index] = row
							train_index += 1
						else:
							test_data[test_index] = row
							test_index += 1
					
					frame_num+=1
				
				#create label vectors
				#boolean -- raining at any time within one hour
				train_dict_keys = list(train_label_dict.keys())
				frame_index = num_train_frames
				while (frame_index < num_files-num_pred_frames):
					i = 0
					result = 0
					while(i < num_pred_frames):
						if train_label_dict[frame_index+i] != 0:
							result=1
							break
						else:
							i+=1

					if year != '2013':
						train_labels.append(result)
					else:
						test_labels.append(result)

					frame_index += 1

	np.save('train_data_'+str(box_width), train_data)
	np.save('train_labels_'+str(box_width), train_labels)
	np.save('test_data_'+str(box_width), test_data)
	np.save('test_labels_'+str(box_width), test_labels)

	print("Rain count: "+str(rain_count))
	print("No rain count: "+str(no_rain))
	print("File count: "+str(total_files))
	print("Rain average for train set: "+str(rain_count/total_files))
	print("Number of test rows added: "+str(test_index))
	print("Number of train rows added: "+str(train_index))


if __name__ == "__main__":
	main()
