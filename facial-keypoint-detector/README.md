[//]: # (Image References)

[image1]: ./images/obamas_with_keypoints.png "Facial Keypoint Detection"

# AIND Term II, Computer Vision Capstone Project
# Facial Keypoint Detection and Real-time Filtering

## Project Overview

Welcome to the Computer Vision capstone project in the AI Nanodegree program! In this project, you’ll combine your knowledge of computer vision techniques and deep learning to build and end-to-end facial keypoint recognition system. Facial keypoints include points around the eyes, nose, and mouth on any face and are used in many applications, from facial tracking to emotion recognition. Your completed code should be able to take in any image containing faces and identify the location of each face and their facial keypoints, as shown below.

![Facial Keypoint Detection][image1]

The project will be broken up into a few main parts in one Python notebook:

__Part 1__ : Investigating OpenCV, pre-processing, and face detection

__Part 2__ : Training a Convolutional Neural Network (CNN) to detect facial keypoints

__Part 3__ : Putting parts 1 and 2 together to identify facial keypoints on any image!

You'll also be given *optional* exercises that allow you to extend this project so that it works on video and allows you to implement fun face filters in real-time!

### Environment

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/udacity/AIND-CV-FacialKeypoints.git
cd AIND-CV-FacialKeypoints
```

2. Create (and activate) a new environment with Python 3.5 and the `numpy` package.

	- __Linux__ or __Mac__:
	```
	conda create --name aind-cv python=3.5 numpy
	source activate aind-cv
	```
	- __Windows__:
	```
	conda create --name aind-cv python=3.5 numpy scipy
	activate aind-cv
	```

3. Install/Update TensorFlow (for this project, you may use CPU only).
	- Option 1: __To install TensorFlow with GPU support__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using the Udacity AMI, you can skip this step and only need to install the `tensorflow-gpu` package:
	```
	pip install tensorflow-gpu==1.1.0
	```
	- Option 2: __To install TensorFlow with CPU support only__:
	```
	pip install tensorflow==1.1.0
	```

4. Install/Update Keras.
 ```
pip install keras -U
```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__:
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	- __Windows__:
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```

6. Install a few required pip packages (including OpenCV).
```
pip install -r requirements.txt
```

### Data

All of the data you'll need to train a neural network is in the AIND-CV-FacialKeypoints repo, in the subdirectory `data`. In this folder are a zipped training and test set of data.

1. Navigate to the data directory
```
cd data
```

2. Unzip the training and test data (in that same location). If you are in Windows, you can download this data and unzip it by double-clicking the zipped files. In Mac, you can use the terminal commands below.
```
unzip training.zip
unzip test.zip
```

You should be left with two `.csv` files of the same name. You may delete the zipped files.

*Troubleshooting*: If you are having trouble unzipping this data, you can download that same training and test data on [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data).

Now, with that data unzipped, you should have everything you need!
