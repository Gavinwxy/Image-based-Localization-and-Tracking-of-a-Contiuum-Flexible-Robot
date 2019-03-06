# Image-based-Localization-and-Tracking-of-a-Contiuum-Flexible-Robot

## Stage 1

Apply classic image processing on images from two camera to:

1. Threshold the shape of the robot
2. Calculate transformation matrix according to template board
3. Register images to construct 3D model of robot (point cloud).

## Stage 2

Use 3D pose from the first stage as training data to train a deep learning model.
 