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
[hog_8]: ./output_images/vehicle_hog_features_8.jpg
[hog_12]: ./output_images/vehicle_hog_features.jpg
[hog_non_vehicle]: ./output_images/non_vehicle_hog_features.jpg
[sliding_window]: ./output_images/sliding_window_result.jpg
[feature_subsampling]: ./output_images/feature_subsample_result.jpg
[heatmap_result]: ./output_images/heat_frames.png
[heatmap_final]: ./output_images/heatmap_output.jpg

####I will consider the [Project Rubric Points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

---

###1.0 Histogram of Oriented Gradients (HOG) Feature Extraction and SVM Classifier

####Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

The code for this step is contained in the section 1.0 of the IPython notebook "pipeline_session.ipynb".

First, I read in all the `vehicle` and `non-vehicle` images from the provided datasets.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][vehicle_non_vehicle]

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) to understand the combination of parameters that will give good result.  Shown below is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_8]

Example, using the `RGB` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_12]

Example of HOG features, for a Non-Vehicle:

![alt text][hog_non_vehicle]

####Deciding on HOG parameters, and other features.

I used the classification accuracy of the SVM classifier to decide on the best parameters for project.

The table below shows the combination of parameters tried and the results of classification accuracy and length of feature vectors obtained. 

| Colour Space 	| Orientation 	| Pixels per cell 	| Cells per block 	| Feature Vector Len | Accuracy |
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

After deciding on using the YCrCb colour space, I also added Spatial binning features and Histogram features to improve the classification. The following table shows the combination of parameters for spatial binning and histogram features and the corresponding accuracies

| Colour Space 	| Spatial size 	| Histogram bins 	| Feature Vector Len | Accuracy |
|:-------------:|:-------------:|:-----------------:|:------------------:|:--------:|
| YCrCb			| - 			| 16				| 4752  			 | 98.48%   |
| YCrCb			| - 			| 32				| 4800  			 | 98.70%   |
| YCrCb			| - 			| 64				| 4896  			 | 98.78%   |
| YCrCb 		| (32, 32) 		| -					| 7776  			 | 98.82%   |
| YCrCb 		| (64, 64) 		| -					| 16992  			 | 99.04%   |
| YCrCb 		| (128, 128) 	| -					| 53856  			 | - 		|
| YCrCb 		| (32, 32) 		| 32				| 7872  			 | 98.99%   |
| **YCrCb** 	| (32, 32) 		| 64				| **7668** 			 | 99.16%   |
| YCrCb 		| (64, 64) 		| 32				| 17088  			 | 99.10%   |
| YCrCb 		| (64, 64) 		| 64				| 17184  			 | 99.18%   |

Since the number of feature vectors affect the run time of the prediction (more feature vector, slower run time), I chose parameters that perform well with lesser feature vector length.

I trained a linear SVM using the Scikit-learn library's function `LinearSVC()`, which is a Linear Support Vector Classification. According to the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), " it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples."

I used a combination of HOG features, Spatial binning and colour histogram features extracted from the dataset as input into the classifier (YCrCb colour space, spatial size of (32,32), histogram bins of 64, HOG orientation of 8, pixels per cell of 8, and cells per block of 2). I made sure to shuffle the dataset and split into train and test sets by using `train_test_split()` function from Scikit-learn Library. See code snippet below.

```python
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
```

Using a test set of 20% of the dataset, The training resulted in a test accuracy of *99.01%*. The result can be seen in section 1.1 of the Ipython notebook.

---

###2.0 Sliding Window Search

####Implement a sliding-window technique and use trained classifier to search for vehicles in images.
<!-- 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows? -->

I used two approaches for the sliding window search technique:

1. In the first approach, I created sub-regions (window) of the image and then classify these sub-regions. If the classification is positive, I then draw a box on this portion of the original image. This procedure is in the section 2.0 and 2.1 of the Ipython notebook. I implemented this on the pipeline by searching different window sizes (`sliding_window()` of `Vehicle` class in `find_vehicles.py`) to make the car detection more robust. However, this implementation is slower than the second approach because the feature extraction is done for every window search.

![alt text][sliding_window]

A false positive classification can be seen in the result from 'sliding window 96x96', and both results show multiple detection. I will treat this issue in the next section.

2. In the second approach, I extract the features only once, then I sub-sampled the features to get all of its overlaying windows. Similar to above, we can also specify a scaling value, which enables us to search different sizes of sub-samples. See Section 2.2 in Ipython notebook for implementation codes.

![alt text][feature_subsampling]


####Using heat maps to remove multiple detection and false positives
<!-- 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier? -->

In order to remove multiple detections and false positives, I implemented heat maps by  adding "heat" (+=1) for all pixels within windows where a positive detection is reported by classifier. I then track predicted windows in a range of consecutive windows, average over the heatmaps and threshold the average of the heatmaps to remove unlikely regions. The function to update heatmaps is shown (the full implementation is in `Vehicle` class in `find_vehicles.py`):

```python
def update_heat(self, current_heat, patience=16):
    """
    Update average heat in frames using n past frames
    """
    self.recent_heats.append(current_heat)
    if len(self.recent_heats) > patience:
        self.recent_heats.pop(0)

    self.ave_heat = np.intc(np.average(self.recent_heats, 0))
```

Here, I show six consecutive frames and their corresponding heatmaps. The heatmaps could be seen to be consistent between following frames:

![alt text][heatmap_result]

The heatmap is in turn used to draw a bounding box around the vehicles using regions of the image that are higher than the heatmap threshold by the `scipy.ndimage.measurements.label()` function. The function identifies individual blobs in the heatmap, each blob then is assumed to be a vehicle and the bounding box is drawn around each blob. Here is an example of bounding boxes from the heatmap identifying two vehicles:

![alt text][heatmap_final]

<!-- 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes. -->

<!-- I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.   -->

<!-- Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7] -->

---
###3.0 Video Implementation

The pipeline was applied on test and project videos in the `Vehicle` class of `find_vehicles.py`.
<!-- 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.) -->

Find in the link below, the final output of the pipeline for the project video

[Project Video Result](https://youtu.be/QYRkMquhtAE)

---

###Discussion
####Insights and observations from the project
<!-- 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? -->

With this project, I am able to better appreciate every detail that goes into technologies that make our lives easier. More can still be done on this project. We can estimate distance of detected vehicles, and/or improve vehicle tracking to identify cars even in occluded situations. This will however, need better image classification algorithms. 

For this solution, real time application is very important. Therefore, it will be very useful to do all these in real-time or at least, very close to real-time.


Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

