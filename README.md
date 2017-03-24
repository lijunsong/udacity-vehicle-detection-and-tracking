**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/v-nv.png
[image2]: ./output_images/rgb-hsv.png
[scale1]: ./output_images/scale15.png
[scale2]: ./output_images/scale2.png
[heatmap1]: ./output_images/hotimage-0.jpg
[heatmap_image1]: ./output_images/hotimagehot-0.jpg
[pipeline_result]: ./output_images/pipeline-result.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This repo contains

 - `utils.py` contains useful helper functions taken from lectures and quiz
 - `train.py` contains the main logic of training and search
 - `vehicle-detection.ipynb` contains the training code
 - `README.md` this writeup.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Training images are loaded in class `CarImages`. Here is an example of vehicle and non-vehicle images.

![vehicle and non-vehicle][image1]


The training is done in `Model` class. HOG feature extraction is done in `preprocess` method. `preprocess` is called from `train` method to extract features from a training image. The actual extraction is done using `skimage.feature.hog`, which takes an image with various parameters and returns the HOG feature vector of the image.


#### 2. Explain how you settled on your final choice of HOG parameters.

To find which color space to apply HOG, I use `get_hog_features` function to return an image equivalent to extracted features, and eyeball how recognizable the image is. Take the following images as example, where we can see that HSV color space has richer HOG features compared with RGB.

![color space explore][image2]

HOG parameters are mostly tuned during sliding window search. I used these parameters

 - orient: 32, because my laptop can't handle large feature vectors, also see Discussion
 - pixels_per_cell: 16, testing shows 16 better than 8
 - cells_per_block: 2


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used HOG features and spatial binning features to train my model. The training is done in `Model` class.

(I used to use a combination of HOG features, color features and spatial features, but those seem only adding more complexity in the code without increasing prediction rate.)

`Model.preprocess` will return a vector containing HOG and spatial features (features are simply concatenated).

`Model.train` will preprocess images and normalize feature vectors to make them zero mean and unit variance. Then features and labels will be shuffled to remove order effect of the training data. Finally 30% training data will be reserved for evaluation before training.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I started with the `slide_window` function (see `utils.py`). This function generates a list of windows to search. Previously, for each window, I extracted features and had the model to make the prediction. Object in different positions will have different size because the further the distance, the smaller the object. The main benefit is that this function fits nicely with previous code. The main drawback was the slow speed because it has to calculate the HOG feature for each window.

Then I adapted the sub-sampling techniques. Windows are generated based on cells and blocks in HOG features. For each image, the technique calculates HOG features only once.

Since windows are generated based on cells and blocks, windows will be moved to next position using metrics based on cells or blocks. If the window moves every 2 cells and a window has 4 cells, the overlap is 50%. Experiment shows that my model works well when the overlap is 75%.

The sub-sampling technique doesn't enlarge the window size, instead it scales the image before feature extraction and keeps window size the same, which is equivalent to zooming the window.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used two scales scanning on YCrCb 3-channel HOG features in the feature vector, which provided decent result.  Here are some images showing scanning process:

Scanning from y 370 to 500 with scaling factor 1

![scaling 1][scale1]

Then scan 400 to 580 with scaling factor of 2

![scaling 2][scale2]

With the sub-sampling technique, the model will extract features from each window and predict whether the extracted piece of image is a car.

Eventually, calling `annotate_cars_in_image(test_image)` will produce an image with a box around cars:

![result][pipeline_result]

I tested different feature vectors and decided to drop the color histogram features to increase training speed. Using only HOG actually has bad performance in my experiement, so I added spatial binning features with reduced the size 16x16x3.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./annotated_P5_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The method `CarSearch.annotate_cars_in_video` does filters and combining boxes.

Class `CarSearch` maintains `history`, a queue of max size 5~10 for potential car positions. The method searches cars in each frame and draws bounding boxes. It then generates a `heatmap` for these bounding boxes, and puts the `heatmap` in the queue.

When the queue is full, a final `heatmap` will be generated and threshed. In my implemented, a pixel continues to live in next frame only when its `heatmap` value is greater than 3 (maximum 7). `scipy.ndimage.measurements.label` will tell how many separated blobs are there in the `heatmap`. With the assumption that each blob is a car, I reconstruct boxes to cover blobs, which removes some temporary false positives.

Here is an example of annotated image and its corresponding **accumulated** `heatmap`.

![heatmap][heatmap1]
![image with box][heatmap_image1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I encountered a problem where the feature vector size was too large to fit into memory, and took forever to train. That feature vector contained HOG features, spatial bins of (32, 32) size and color maps, length around 20k in total. I then reduced HOG size by increasing the orient parameter, remove color histograms, and reduce spatial bins size to (16,16). With these improvement, it can produce decent result. (I can also use generator to solve memory issue, but the slowness is still a problem.)

The scaling factor in sliding window search is tricky. Even if the model has 98% accuracy, if the scaling factor increase to 3, or decrease to 1.5, the model is going to have a lot of false positives. To solve this problem, I need explore more to get a better model and data. I believe the 98% accuracy on test set is overfitting because we have similar images in the given data.

I noticed that current reconstruction of boxes will combine two cars nearby. Current algorithm will definitely fail in this situation. One way to solve this problem is to filter out more unqualified heatmap to produce tighter boxes, or probably introduce prediction confidence level to increment heatmap map (instead of +1)

Another situation this algorithm will fail is when the car is in bit slop where the start and end position (y-axis) of sliding window is not tuned for these situations.
