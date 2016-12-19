import h5py
import numpy as np
import tarfile, sys
import os
import gzip
import re

'''
uses 2TB of publicly available Iowa rain data
initially  represented as a 1057x1741 matrix, with each cell representing mm/hr rainfall
matrices every 5 minutes from 2009-2013
extracts double-compressed data into 'extracted_dir'
constructs HDF5 file organized by year/month/time_frame for further pre-processing

'radius' var controls width around point of prediction (Ames, IA -- [602, 845] in matrix)
'''

compress_dir = "/Shared/bdagroup2/RainfallIowa"
extracted_dir = "/Users/lfranzen/bda/extracted_data"
radius = 50

def untar(directory):
	for fname in os.listdir(directory):
		if os.path.isdir(directory+"/"+fname) and not fname.startswith('.'):
			for fname2 in os.listdir(directory+"/"+fname):
				if (fname2.endswith("tar")):
					tar = tarfile.open(directory+"/"+fname+"/"+fname2)
					tar.extractall(extracted_dir)
					tar.close()
					print ("Extracted tar data to: " + extracted_dir)
				else:
					print ("Not a tar file: '%s '" % sys.argv[0])
		

def main():
	untar(compress_dir)

	file_regex = "([0-9]{2}[A-Z]{3}[0-9]{4}_[0-9]{6})"
	year_regex = "([0-9]{4})"
	month_regex = "([A-Z]{3})"
	python_regex = "(_\w+)"

	box_width = radius*2+1
	hf = h5py.File("data_"+str(box_width)+"_all.hdf5", "w")

	for fname in os.listdir(extracted_dir):
		with gzip.open(extracted_dir+"/"+fname, 'rt') as f:
			rain_data = np.zeros((box_width, box_width))
			rain_data_index = 0
			for index, line in enumerate(f):
				if index < 601-radius: #index to control radius around Ames
					continue
				elif index <= 601+radius: #index to control radius around Ames
					line_arr = np.array(line.strip().split())
					rain_data[rain_data_index:] = line_arr[844-radius:844+radius+1:1]
					rain_data_index += 1
				else:
					break

			data_name = re.search(file_regex, fname, re.IGNORECASE).group()
			year = re.search(year_regex, data_name).group()
			month = re.search(month_regex, data_name, re.IGNORECASE).group()
			hdf_name = year+"/"+month+"/"+data_name


			hf.create_dataset(hdf_name, data=rain_data)

if __name__ == "__main__":
	main()
