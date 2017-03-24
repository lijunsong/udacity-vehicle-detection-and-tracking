##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This repo contains

 - `utils.py` contains useful helper functions taken from lectures and quiz
 - `train.py` contains the main logic of training and search
 - `vehicle-detection.ipynb` contains the training code
 - `writeup_report.md` this writeup.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Training images are load in class `CarImages`. Here is an example of vehicle and non-vehicle images.

TODO images


The training is done in `Model` class where HOG feature extraction is done in `preprocess` method. `preprocess` is called from `train` method to extract features from a training image. The actual extraction is done using `skimage.feature.hog`, which takes an image with various parameters and returns the HOG feature vector of the image.


#### 2. Explain how you settled on your final choice of HOG parameters.

To find out a set of good parameters, I have the `hog` function to return a featured image to feel how recognizable the feature is. Take the following images as example, where we can see that TODO color space is a better choice.

TODO images


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used a combination of HOG features, color features and spatial features. The training is done in `Model` class.

`Model.preprocess` will return a vector containing HOG, color and spatial features (features are simply concatenated).

`Model.train` will preprocess images and normalize feature vectors to make them zero mean and unit deviation. Then features and labels will be shuffled to remove order effect in the training data. Finally 30% training data will be reserved for evaluation before training.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started with the `slide_window` function. The function generates a list of windows to search. Previously for each window, I extracted features (HOG, spatial and color) and had the model to make the prediction. Windows in different positions will have different size because the further the distance, the smaller the car. The main benefit is that this function fits nicely with previous code. The main drawback was the slow speed because it has to calculate the HOG feature for each window.

Then I adapted the sub-sampling techniques. Windows are generated based on cells and blocks in HOG features while spatial and color features are calculated based on that.  For each image, the technique calculates HOG features only once.

Since windows are generated based on cells and blocks, windows will be moved according to the number of cells or blocks. If the window moves every 2 cells and a window has 16 cells, the overlap is 14/16. The sub-sampling technique doesn't enlarge the window size, instead it scales the image before feature extraction and keeps window size the same.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

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
