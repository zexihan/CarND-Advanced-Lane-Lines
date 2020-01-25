# Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/1_distortion_correction.png "Undistorted"
[image2]: ./output_images/2_road_undistorted.jpg "Road Transformed"
[image3]: ./output_images/3_thresholded_binary.png "Binary Example"
[image4-1]: ./output_images/4_perspective_transform_straight.png "Warp Example 1"
[image4-2]: ./output_images/4_perspective_transform_curved.png "Warp Example 2"
[image5]: ./output_images/5_lane_boundary.jpg "Fit Visual"
[image6]: ./output_images/6_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step and all the following steps is contained in the IPython notebook located in **"./P2.ipynb"**.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

`cv2.undistort()` function takes in a distorted image, our camera matrix, and distortion coefficients. And it returns an undistorted image. An undistorted test image is shown below:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. A trapezoidal shape region of interest mask is also applied here to obtain a cleaner road binary image. The code for this part includes two functions called `region_of_interest()` and `create_thresholded_binary()` Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
# For source points I'm picking four points in a trapezoidal shape 
# that would represent a rectangle when looking down on the road from above.
src = np.float32([[677, 443], [1110, 719], [189, 719], [601, 443]])
# For destination points, I'm arbitrarily choosing some points to be
# a nice fit for displaying our warped rectangle road
offset_warp = 350 # offset of lane lines to warped image side edges for dst points
dst = np.float32([[img_size[0] - offset_warp, 0],
                  [img_size[0] - offset_warp, img_size[1] - 1],
                  [offset_warp, img_size[1] - 1],
                  [offset_warp, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 677, 443      | 930, 0        | 
| 1110, 719     | 930, 719      |
| 189, 719      | 350, 719      |
| 601, 443      | 350, 0        |

I verified that my perspective transform was working as expected by drawing the shape connected by the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4-1]

![alt text][image4-2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I plot a histogram of where the binary activations along all the columns in the lower half of the image. With this histogram I am adding up the pixel values along each column in the image. I use the two highest peaks from my histogram as a starting point for determining where the lane lines are, and then use sliding windows moving upward in the image (further along the road) to determine where the lane lines go. And finally I can fit my lane lines with a 2nd order polynomial. The functions `find_lane_pixels()` and `fit_polynomial()` complete the above tasks in this part. The polynomial coefficients of the left and right lane lines are listed below:

[-5.43259788e-04  7.75639759e-01  1.40956587e+02]

[-3.87759776e-04  6.80371442e-01  6.94087219e+02]

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented the calculation of the radius of curvature and vehicle position with respect to center with the `measure_curvature_real()` and `measure_vehicle_position()` functions. The formula of calculating the radius of curvature is listed in an [awesome tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php). I also derived a conversion from pixel space to world space in my images, compare your images with U.S. regulations that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each. 

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/580 # meters per pixel in x dimension
```

The vehicle position is calculated with the lane_center and image_center as below:

```python
# Assume the camera is mounted at the center of the car
image_center = img_size[0] // 2
    
# Use the bottom one fifth of the image to locate car center
mean_left = np.mean(left_fitx[-left_fitx.shape[0] // 5:])
mean_right = np.mean(right_fitx[-right_fitx.shape[0] // 5:])
lane_center = (mean_right + mean_left) / 2

line_base_pos = (lane_center - image_center) * xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in in the functions `warp_lane_boundaries()` and `put_text()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/zexihan/CarND-Advanced-Lane-Lines/blob/master/videos_output/project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

To run my algorithms on the video stream, I applied tracking, sanity check, reset, and smoothing to make my pipeline more robust. A `Line()` class is defined to keep track of some important parameters measured from frame to frame. `recent_xfitted` (x values of the last n fits of the line) is tracked over the last n frames, the mean value of which is used as the current `fitx` for smoothing. Lane width is also tracked and checked in each frame to ensure that there is no abrupt change to it.

However, the tracking is not implemented for the first n frames in the video. As a result, the pipeline will fail if the video has a challenging start.