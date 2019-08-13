import cv2
import numpy as np


def findCorrespondPts(ptsL, ptsR, F, thresh = 0.001):

    #Load points in segment results
    ptsL = ptsL.T.astype('float32')
    ptsR = ptsR.T.astype('float32')

    # Calculate epipolar lines based on fundamental matrix
    lines = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, F)
    lines = lines.reshape(-1,3)
    # Obtain all matched points in right image
    # By calculating distance of point to line
    thresh = thresh
    bias = np.ones((len(ptsR), 1))
    ptsR_bias = np.hstack((ptsR, bias))
    # Point to line distance calculation (not normalized)
    ptLineRes = np.abs(np.matmul(lines, ptsR_bias.T))
    # All the corresponding poinst found by index
    min_idx = np.argmin(ptLineRes, axis=1)
    min_arg = np.min(ptLineRes, axis=1)
    # Filter them by distance thresh
    th_idx = np.where(min_arg < thresh)
    ft_idx = np.take(min_idx, th_idx)
    ptsL_cors = np.take(ptsL, th_idx, axis = 0)
    ptsR_cors = np.take(ptsR, ft_idx, axis = 0)
    
    return ptsL_cors, ptsR_cors



