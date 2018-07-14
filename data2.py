# Import necessary libraries
from keras.datasets import mnist
import pandas as pd
import numpy as np
import cv2 as cv
import os

# Get data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Resize Ys to 1-D array
Y_train.resize((Y_train.shape[0], 1))
Y_test.resize((Y_test.shape[0], 1))

# Stack Xs and Ys vertically
X = np.vstack((X_train, X_test))
Y = np.vstack((Y_train, Y_test))


def save(rel_path, X, y, start_id):
	"""
	Save MNIST data as images. The name of the image follows 
	the rule given below-
	               <ID>_<Label>.png
	
	rel_path: path where images will be saved relative to the
	          current directory.
	X: Image array.
	y: Label array.
	start_id: Starting id of images.
	
	RETURNS: Last id. 
	"""
	id_num = start_id

	# If absolute path does not exist, make it.
	abs_path = os.path.join(os.getcwd(), rel_path)
	if not os.path.exists(abs_path):
		os.makedirs(abs_path)

	# Iterate through images and save them.
	for row in range(X.shape[0]):
	    img_array = X[row, :].reshape((28, 28))
	    
	    # Image Id
	    img_id = str(id_num) + '_' + str(y[row][0])

	    # Image name.
	    img_name = img_id + '.png'
	    
	    # Write image
	    cv.imwrite(os.path.join(abs_path, img_name), img_array)
	    
        # increment Id number
	    id_num += 1

	# Return last Id
	return id_num - 1

# Save testing and training images in data2 folder
id_num = save('data2', X, Y, 1)

# Print the number of files written to disk
print('Total files written:', id_num)