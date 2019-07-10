import cv2
import numpy as np


def loadPts(segRes):
    '''Return Nx2 points from segmentation result
    '''
    # A conversion on coordinates is necessary
    # From mat to image coords
    coordy, coordx = np.where(segRes == 255)
    pts = np.vstack((coordx, coordy))
    return pts.T



def findCorrespondPts(segRes1, segRes2, F, sampleSize = 200, thresh = 0.5):
    # be careful about coordinate representation
    sample_num = sampleSize

    #Load points in segment results
    ptsL = loadPts(segRes1).astype('float32')
    ptsR = loadPts(segRes2).astype('float32')

    pts_dict_L = {}
    ptsL_sample = []

    #Sample on Left image
    xL, yL = ptsL.T[0], ptsL.T[1]
    for x, y in zip(xL, yL):
        pts_dict_L[y] = x

    if sample_num > len(xL):
        print('Sample number exceeed total points number!')
        return None
    else:
        sample_idx = np.floor(np.linspace(min(yL), max(yL), num = sample_num))
        for ele in sample_idx:
            if ele in pts_dict_L:
                ptsL_sample.append([pts_dict_L[ele], ele])
        ptsL_sample = np.array(ptsL_sample).astype('float32')


    # Calculate epipolar lines based on fundamental matrix
    lines = cv2.computeCorrespondEpilines(ptsL_sample.reshape(-1, 1, 2), 1, F)
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
    ptsL_cors = np.take(ptsL_sample, th_idx, axis = 0)
    ptsR_cors = np.take(ptsR, ft_idx, axis = 0)


    return ptsL_cors, ptsR_cors

