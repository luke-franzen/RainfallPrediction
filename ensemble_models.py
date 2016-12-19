import numpy as np
import h5py
from sklearn.metrics import confusion_matrix

#location of all files
file_loc = "/Users/Franzen/Desktop/BigDataAnalytics/project/test/onehr/"
matrix_size = ["49","75","101"]

#location of true class labels
y_train = np.load("/Users/Franzen/Desktop/BigDataAnalytics/project/test/onehr/train_halflabels_49.npy")
y_test = np.load("/Users/Franzen/Desktop/BigDataAnalytics/project/test/onehr/test_halflabels_49.npy")

#threshold of prediction (used to bin real-number predicted outputs from 'build_model.py' as binary)
threshold = .012

def main():
	ens_train = np.zeros((280628,1), np.int8)
	ens_test = np.zeros((70341,1), np.int8)

	for data in matrix_size:
		for neurons in range(100,2100,100):
			train_pred_hdf = h5py.File(file_loc+data+"x"+data+"/trainpreds_"+data+"_"+str(neurons)+"neurons.hdf5", 'r')
			test_pred_hdf = h5py.File(file_loc+data+"x"+data+"/testpreds_"+data+"_"+str(neurons)+"neurons.hdf5", 'r')

			train_preds = np.array(train_pred_hdf['data'])
			test_preds = np.array(test_pred_hdf['data'])

			ens_train = ens_train+train_preds
			ens_test = ens_test+test_preds

	ens_train = ens_train/(len(matrix_size)*20)
	ens_test = ens_test/(len(matrix_size)*20)

	ens_train[ens_train <= threshold] = 0
	ens_train[ens_train > threshold] = 1
	ens_test[ens_test <= threshold] = 0
	ens_test[ens_test > threshold] = 1

	train_confusion = confusion_matrix(y_train, ens_train)
	test_confusion = confusion_matrix(y_test, ens_test)

	train_acc = (train_confusion[0][0]+train_confusion[1][1])/(train_confusion[0][0]+train_confusion[0][1]+train_confusion[1][0]+train_confusion[1][1])
	train_err = 1 - train_acc
	train_sens = (train_confusion[1][1])/(train_confusion[1][1]+train_confusion[1][0])
	train_spec = (train_confusion[0][0]/(train_confusion[0][0]+train_confusion[0][1]))

	test_acc = (test_confusion[0][0]+test_confusion[1][1])/(test_confusion[0][0]+test_confusion[0][1]+test_confusion[1][0]+test_confusion[1][1])
	test_err = 1 - test_acc
	test_sens = (test_confusion[1][1])/(test_confusion[1][1]+test_confusion[1][0])
	test_spec = (test_confusion[0][0])/(test_confusion[0][0]+test_confusion[0][1])

	print("\n")
	print("Dataset: "+data+"x"+data)
	print("Threshold: "+str(threshold))
	print("Train Performance")
	print("-----------------")
	print("Accuracy is: "+str(train_acc))
	print("Error is: "+str(train_err))
	print("Sensitivity: "+str(train_sens))
	print("Specificity: "+str(train_spec))
	print(train_confusion)

	print("\n")
	print("Test Performance")
	print("----------------")
	print("Accuracy is: "+str(test_acc))
	print("Error is: "+str(test_err))
	print("Sensitivity: "+str(test_sens))
	print("Specificity: "+str(test_spec))
	print(test_confusion)
	print("\n")

if __name__ == "__main__":
    main()