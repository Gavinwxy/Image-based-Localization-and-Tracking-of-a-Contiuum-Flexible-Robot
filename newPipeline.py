import numpy as np 
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures


def ransacCurveFit_v2(seg_img, degree = 7, trials = 100, sampleNum = 100):
# Find all the data points in mask
    data = np.where(seg_img==255)
    X,y = data[0], data[1]
    X = X.reshape(-1,1)

    # Create poly feature to fit
    poly = PolynomialFeatures(degree = degree, include_bias = True)
    X = poly.fit_transform(X)

    # Create RANSAC model
    ransac = RANSACRegressor(min_samples=0.3, max_trials=trials)
    ransac.fit(X,y)

    # Prepare to plot the curve
    low = 55
    upper = max(X[:,1])

    point_num = sampleNum


    x_sample = np.linspace(low, upper, point_num)
    x_sample_trans = poly.fit_transform(x_sample.reshape(-1,1))
    y_sample = ransac.predict(x_sample_trans)
    
    pts = np.vstack((y_sample, x_sample))
    return pts


def findCorrespondPts_v2(ptsL, ptsR, F, thresh = 0.001):

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