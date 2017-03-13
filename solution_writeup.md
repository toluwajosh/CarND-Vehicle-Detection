##Vehicle Detection and Tracking
###(Udacity Nanodegree Project 5)

---

<!-- **Vehicle Detection Project** -->

The following are the goals and steps of this project:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. *Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.*
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[vehicle_non_vehicle]: ./output_images/vehicle_non_vehicle.jpg
[hog_8]: ./output_images/HOG_example.jpg
[hog_12]: ./output_images/sliding_windows.jpg
[hog_non_vehicle]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

####I will consider the [Project Rubric Points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

---

###1.0 Histogram of Oriented Gradients (HOG) Feature Extraction and SVM Classifier

####Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

The code for this step is contained in the section 1.0 of the IPython notebook 'pipeline_session.ipynb'.

First, I read in all the `vehicle` and `non-vehicle` images from the provided datasets.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][vehicle_non_vehicle]

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  Shown below is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][hog_8]

Example using the `RGB` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_12]

Example of HOG features, for a Non-Vehicle:

![alt text][hog_non_vehicle]

####Deciding on HOG parameters, and other features.

I used the classification accuracy of the SVM classifier to decide on the best parameters for project. (Experiment was run on a computer with Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz 16.0GB RAM)

The table below shows the combination of parameters tried and results of classification accuracy and run time. 

| Colour Space 	| Oreintation 	| Pixels per cell 	| Cells per block 	| Feature Vector Len | Accuracy |
|:-------------:|:-------------:|:-----------------:|:-----------------:|:------------------:|:--------:|
| RGB 			| 8 			| 8					| 2 				| 2400  			 | 95.44%   |
| RGB 			| 10 			| 8					| 2 				| 5880  			 | 95.78%   |
| RGB 			| 12 			| 8					| 2 				| 7056  			 | 96.79%   |
| HLS 			| 8 			| 8					| 2 				| 2400  			 | 97.94%   |
| HLS 			| 10 			| 8					| 2 				| 5880  			 | 98.23%   |
| HLS			| 12 			| 8					| 2 				| 7056  			 | 98.23%   |
| YCrCb			| 8 			| 6					| 2 				| 7776  			 | 97.78%   |
| YCrCb 		| 8 			| 8					| 1 				| 1536  			 | 97.72%   |
| **YCrCb**		| 8 			| 8					| 2 				| 4704  			 | 98.37%   |
| YCrCb 		| 8 			| 8					| 3 				| 7776  			 | 98.25%   |
| YCrCb			| 8 			| 10				| 2 				| 2400  			 | 97.18%   |
| YCrCb			| 10 			| 8					| 2 				| 5880  			 | 98.34%   |
| YCrCb			| 12 			| 8					| 2 				| 7056  			 | 98.45%   |

After deciding on using the YCrCb colour space, I also added Spatial binning features and Histogram features to improve the classification. The following table shows the combination of parameters for spatial binning and histogram features.

| Colour Space 	| Spatial size 	| Histogram bins 	| Feature Vector Len | Accuracy |
|:-------------:|:-------------:|:-----------------:|:------------------:|:--------:|
| YCrCb			| - 			| 16				| 4752  			 | 98.48%   |
| YCrCb			| - 			| 32				| 4800  			 | 98.70%   |
| YCrCb			| - 			| 64				| 4896  			 | 98.78%   |
| YCrCb 		| (32, 32) 		| -					| 7776  			 | 98.82%   |
| YCrCb 		| (64, 64) 		| -					| 16992  			 | 99.04%   |
| YCrCb 		| (128, 128) 	| -					| 53856  			 | - 		|
| YCrCb 		| (32, 32) 		| 32				| 7872  			 | 98.99%   |
| **YCrCb** 	| (32, 32) 		| 64				| 7668  			 | 99.16%   |
| YCrCb 		| (64, 64) 		| 32				| 17088  			 | 99.10%   |
| YCrCb 		| (64, 64) 		| 64				| 17184  			 | 99.18%   |


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

