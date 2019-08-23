# Image based Localization and Tracking of a Continuum Flexible Robot

## Intro

In this project,  a vision-based system is designed to achieve tracking and shape estimation of Concentric Tube Robot (CTR) based on binocular stereo images.  The system consists of two main stages which are segmentation and 3D reconstruction. The CTR is firstly segmented from stereo image pair by background removal techniques.  Then epipolar geometry constraints in binocular stereo vision are applied to reconstruct robot 3D shape.  The structure of CTR is shown in figure below.



<img src="./intro_fig/CTR.JPG" width="600">

## Dependencies

- Python 3.7
- OpenCV 3.4
- Scikit-learn

## Dataset

Dataset contains 800 image pairs taken by two camera from different views. Ground truth label are given as the length of each tube. Information of grid template is also given. The circle diameter is 6.8 mm and distance between two adjacent dots is 18.2 mm. An example of two view images are shown below.

**Left view image**

<img src="./intro_fig/cam2_0.png" width="500">

**Right view image**

<img src="./intro_fig/cam1_0.png" width="500">



## Single Camera Calibration

Calibration is done using Zhang's method which is integrated in OpenCV library. Circle locations are obtained by blob detector. Chessboard plots are shown below. 

​      <img src="./intro_fig/chessboard_left.png" width="350">             <img src="./intro_fig/chessboard_right.png" width="350"> 



## Image Segmentation

1. Original Image

   <img src="./intro_fig/cam1_47.png" width="500">

2. Results from Otsu Thresholding followed by morphological operation

   <img src="./intro_fig/otsu_binary.jpg" width="500">

3. Post-processing by RANSAC with polynomial regression of degree 7

   <img src="./intro_fig/post_process.png" width="500">



## Geometric Constraints Calculation

1. Collecting corresponding points by prior knowledge. Here we label ending points and joint points as reliable correspondences.

   <img src="./intro_fig/corpts.JPG" width="800">

   

2. Recovered camera transformations

   

   <img src="./intro_fig/camera_setting.JPG" width="600">

3. Verification on epipolar lines

   <img src="./intro_fig/epilinePlotL.jpg" width="500">

   <img src="./intro_fig/epilinePlotR.jpg" width="500">

## Triangulation

Apply direct linear transformation (DLT) based on obtained geometric relation.

<img src="./intro_fig/res.JPG" width="500">

