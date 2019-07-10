import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import time
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def imshow(img, size=None):
    if size == None:
        plt.figure(figsize=(14,18))
    else:
        plt.figure(figsize=size)
    plt.imshow(img)
    plt.show()

def imshow2(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def backgroundExtract(frames):
    avgR, avgG, avgB = None, None, None
    fcnt = 0
    for frame in frames:
        if frame is None:
            break
        frame = cv2.imread(frame)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        B, G, R = cv2.split(frame.astype("float"))
        if avgR is None:
            avgR = R
            avgG = G
            avgB = B
        else:
            avgB = avgB + 1/fcnt*(B - avgB)
            avgG = avgG + 1/fcnt*(G - avgG)
            avgR = avgR + 1/fcnt*(R - avgR)
        
        fcnt += 1

        avg = cv2.merge([avgB, avgG, avgR]).astype('uint8')
        #avg = cv2.cvtColor(avg, cv2.COLOR_HSV2BGR)

    return avg

def robotSegment(img, bg, th):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype('float32')

    #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)

    thresh = th

    sub = np.abs(img_gray - bg_gray)
    sub = (sub>thresh)*255
    sub = sub.astype('uint8')

    kernel1 = np.ones((2, 2), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #erosion = cv2.erode(sub,kernel,iterations = 1)

    # Openning operation to remove noise
    res = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel1)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel2)

    return res


def robotSegmentv2(img, bg):

    # Color Threshold
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([45, 70, 70])
    upper_bound = np.array([75, 255, 255])

    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    mask_inv = cv2.bitwise_not(mask)

    img_mask = cv2.bitwise_and(img, img, mask=mask_inv)
    bg_mask = cv2.bitwise_and(bg, bg, mask=mask_inv)

    img_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY).astype('float32')
    bg_gray = cv2.cvtColor(bg_mask, cv2.COLOR_BGR2GRAY).astype('float32')

    img_gray = cv2.GaussianBlur(img_gray,(5,5),0)
    bg_gray = cv2.GaussianBlur(bg_gray,(5,5),0)

    thresh = 25
    
    sub = np.abs(img_gray - bg_gray)
    #sub1 = np.abs(img1[:,:,0] - bg1[:,:,0])
    #sub2 = np.abs(img1[:,:,1] - bg1[:,:,1])
    #sub3 = np.abs(img1[:,:,2] - bg1[:,:,2])
    #sub = (sub1+sub2+sub3)/3
    sub = (sub>thresh)*255
    sub = sub.astype('uint8')

    kernel1 = np.ones((2, 2), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #erosion = cv2.erode(sub,kernel,iterations = 1)

    # Openning operation to remove noise
    e = cv2.morphologyEx(sub, cv2.MORPH_ERODE, kernel1)
    res = cv2.morphologyEx(e, cv2.MORPH_DILATE, kernel2)

    return res

def downsample(img, sample_interval = 10):

    x, y = np.where(img == 255)
    data = []
    for x_loc, y_loc in zip(x, y):
        data.append([x_loc, y_loc])

    sample_interval = sample_interval
    sample_bnd = np.floor(len(data)/sample_interval).astype('int')
    sample_idx = [i*sample_interval for i in range(0, sample_bnd)]
    sample_data = [data[i] for i in sample_idx]

    sample_img = np.zeros(img.shape[:2]).astype('int')
    for x, y in sample_data:
        sample_img[x][y] = 255

    return img

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()



def ransacCurveFit(seg_img, degree = 7, trials = 100):
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
    #low = min(X[:,1])
    low = 55
    upper = max(X[:,1])
    point_num = upper - low + 1

    x_plot = np.linspace(low, upper, point_num).astype('int')
    x_plot_t = poly.fit_transform(x_plot.reshape(-1,1))
    y_plot = ransac.predict(x_plot_t).astype('int')

    curve_fit = np.zeros(seg_img.shape)
    for x, y in zip(x_plot, y_plot):
        if x < upper:
            curve_fit[x,y] = 255

    return curve_fit