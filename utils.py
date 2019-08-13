
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def drawPoints(img, pts, colors):
    for pt, color in zip(pts, colors):
        cv2.circle(img, tuple(pt), 2, color, -1)

def drawLines(img, lines, colors):
    _, c, _ = img.shape
    for r, color in zip(lines, colors):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img, (x0, y0), (x1, y1), color, 1)

def rectifyPlot(imgl_rect, imgr_rect, lineNum):
    img_size = imgl_rect.shape[:2][::-1]
    rectified = np.hstack((imgl_rect, imgr_rect))

    lineThickness = 1
    lineColor = [0, 255, 0]
    numLines = lineNum
    interv = round(img_size[0] / numLines)
    x1 = np.zeros((numLines, 1))
    y1 = np.zeros((numLines, 1))
    x2 = np.full((numLines, 1), (4*img_size[1]))
    y2 = np.zeros((numLines, 1))
    for i in range(0, numLines):
        y1[i] = i * interv
    y2 = y1

    for i in range(0, numLines):
        cv2.line(rectified, (x1[i], y1[i]), (x2[i], y2[i]), lineColor, lineThickness)
        
    return rectified 

def epilinePlot(pts1, img1, img2, F, view = 1):
    '''Notice that img2 should be undistorted image and pts1 is based on undistorted image
    args:
        pts1: numpy.array Nx2 points in first image
    '''
    ptNum = pts1.shape[0]
    color = []
    for i in range(ptNum):
        color.append(tuple(np.random.randint(0,255,3).tolist())) 

    drawPoints(img1, pts1, color)  
    # find epilines corresponding to points in left image and draw them on the right image
    epilinesL = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), view, F)
    epilinesL = epilinesL.reshape(-1, 3)
    drawLines(img2, epilinesL, color)

    return img1, img2


def Projection3D(pts3D):
    Xs = pts3D[0]
    Ys = pts3D[1]
    Zs = pts3D[2]

    #Ys = res3D[0]
    #Zs = res3D[1]
    #Xs = res3D[2]

    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(Xs, Ys, Zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Reconstructed Robot Shape')
    plt.show()


def distCal(pts):
    L = 0
    Xs, Ys, Zs = pts[0], pts[1], pts[2]
    for idx in range(len(pts)-1):
        xd = (Xs[idx-1]-Xs[idx])**2
        yd = (Ys[idx-1]-Ys[idx])**2
        zd = (Zs[idx-1]-Zs[idx])**2
        L += np.sqrt(xd+yd+zd)
    return L