import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from utils import imshow, imshow2



# Construct Object Points
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((36, 3), np.float32)
objp[0]  = (0, 0, 0)
objp[1]  = (0, 2, 0)
objp[2]  = (0, 4, 0)
objp[3]  = (0, 6, 0)
objp[4]  = (1, 1, 0)
objp[5]  = (1, 3, 0)
objp[6]  = (1, 5, 0)
objp[7]  = (1, 7, 0)
objp[8]  = (2, 0, 0)
objp[9]  = (2, 2, 0)
objp[10] = (2, 4, 0)
objp[11] = (2, 6, 0)
objp[12] = (3, 1, 0)
objp[13] = (3, 3, 0)
objp[14] = (3, 5, 0)
objp[15] = (3, 7, 0)
objp[16] = (4, 0, 0)
objp[17] = (4, 2, 0)
objp[18] = (4, 4, 0)
objp[19] = (4, 6, 0)
objp[20] = (5, 1, 0)
objp[21] = (5, 3, 0)
objp[22] = (5, 5, 0)
objp[23] = (5, 7, 0)
objp[24] = (6, 0, 0)
objp[25] = (6, 2, 0)
objp[26] = (6, 4, 0)
objp[27] = (6, 6, 0)
objp[28] = (7, 1, 0)
objp[29] = (7, 3, 0)
objp[30] = (7, 5, 0)
objp[31] = (7, 7, 0)
objp[32] = (8, 0, 0)
objp[33] = (8, 2, 0)
objp[34] = (8, 4, 0)
objp[35] = (8, 6, 0)



# Define Blob Detector

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 700 #1000   # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.5

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)


# Blob Detector for accurate point detection (left view)
img_left = cv2.imread('cam2_11.png')

gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
keypoints = blobDetector.detect(gray) # Detect blobs.

points = {}
for keypoint in keypoints:
    points[keypoint.pt[0]] = keypoint.pt[1]

coordx = [key for key in points.keys()]
coordx.sort(reverse=True)

acc_coords = []
for x in coordx:
    acc_coords.append([[x, points[x]]])

acc_coords = np.array(acc_coords)
acc_coords_left = acc_coords.astype('float32')
#np.save('blob_loc_left.npy', acc_coords_left)


imgs = glob.glob('./data_calib/cam2/*.png')

img_points = []
obj_points = []
for frame in imgs:
    obj_points.append(objp)
    img = cv2.imread(frame)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = blobDetector.detect(gray) # Detect blobs.

    points = {}
    for keypoint in keypoints:
        points[keypoint.pt[0]] = keypoint.pt[1]

    coordx = [key for key in points.keys()]
    coordx.sort(reverse=True)

    acc_coords = []
    for x in coordx:
        acc_coords.append([[x, points[x]]]) 
    img_points.append(acc_coords)
    
    acc_coords = np.array(acc_coords).astype('float32')
    #
    im_with_keypoints = cv2.drawChessboardCorners(img, (4,9), acc_coords, True)
    cv2.imshow('img', im_with_keypoints)
    cv2.waitKey(5000)
    #
    
cv2.destroyAllWindows()
img_points_left = np.array(img_points).astype('float32')


# Blob Detector for accurate point detection (right view)
# Points in the right view
img_right = cv2.imread('cam1_14.png')

gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
keypoints = blobDetector.detect(gray) # Detect blobs.

points = {}
for keypoint in keypoints:
    points[keypoint.pt[0]] = keypoint.pt[1]

coordx = [key for key in points.keys()]
coordx.sort(reverse=True)

acc_coords = []
for x in coordx:
    acc_coords.append([[x, points[x]]])

acc_coords = np.array(acc_coords)
acc_coords_right = acc_coords.astype('float32')
#np.save('blob_loc_right.npy', acc_coords_right)


imgs = glob.glob('./data_calib/cam1/*.png')

img_points = []
obj_points = []
for frame in imgs:
    obj_points.append(objp)
    img = cv2.imread(frame)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = blobDetector.detect(gray) # Detect blobs.

    points = {}
    for keypoint in keypoints:
        points[keypoint.pt[0]] = keypoint.pt[1]

    coordx = [key for key in points.keys()]
    coordx.sort(reverse=True)
    #coordx.sort()
    acc_coords = []
    for x in coordx:
        acc_coords.append([[x, points[x]]]) 
    img_points.append(acc_coords)
    
    acc_coords = np.array(acc_coords).astype('float32')
    
    #
    im_with_keypoints = cv2.drawChessboardCorners(img, (4,9), acc_coords, True)
    cv2.imshow('img', im_with_keypoints)
    cv2.waitKey(5000)
    #
    
cv2.destroyAllWindows()
img_points_right = np.array(img_points).astype('float32')

# Left view calibration result
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points_left, gray.shape[::-1], None, None)

data = {'ret': ret, 'intr_mat': mtx, 'distortion_coeff': dist, 'R': rvecs, 'T': tvecs}

#np.save('calibration_coeff_left.npy', data)

# right view calibration result
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points_right, gray.shape[::-1], None, None)

data = {'ret': ret, 'intr_mat': mtx, 'distortion_coeff': dist, 'R': rvecs, 'T': tvecs}

#np.save('calibration_coeff_right.npy', data)