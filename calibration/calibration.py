import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import glob



def calibrate(images, template_size, scale = 1):
    H, W = template_size
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

    objp = np.zeros((H*W, 3), np.float32)
    objp[:, :2] = np.mgrid[0:H, 0:W].T.reshape(-1, 2)*scale

    objpoints = [] 
    imgpoints = [] 
    cal_images = {}

    for idx, image in enumerate(images):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (H,W), None)

        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (H,W), corners2,ret)
            cal_images[idx] = image

            #cv2.imshow('img',img)
            #cv2.waitKey(500)

    #cv2.destroyAllWindows()
    
    img_shape = gray.shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    calibration = {
    'objpoints': objpoints,
    'imgpoints': imgpoints,
    'cal_images': cal_images,
    'mtx': mtx,
    'dist': dist,
    'rvecs': rvecs,
    'tvecs': tvecs
    }

    return calibration 


class CameraCalibrator:
    def __init__(self, img_dir1 = None, img_dir2 = None, template_size = None, scale = 1, img_size = None):
        self.cam1_calib_status = False
        self.cam2_calib_status = False
        self.stereo_calib_status = False
        self.template_size = template_size
        self.scale = scale
        self.img_size = img_size
        # Single camera calibration Loading
        self.imgs1 = img_dir1
        
        # Binocular stereo camera calibration Loading
        if img_dir2 is not None:
            self.imgs2 = img_dir2
        
        self.params1 = None
        self.params2 = None
        self.stereo_params = None

    def calibrate_process(self, stereo = False):
        self.params1 = calibrate(self.imgs1, self.template_size, self.scale)
        self.cam1_calib_status = True
        self.params2 = calibrate(self.imgs2, self.template_size, self.scale)
        self.cam2_calib_status = True

        if stereo:
            objpts1, imgpts1 = self.params1['objpoints'], self.params1['imgpoints']
            objpts2, imgpts2 = self.params2['objpoints'], self.params2['imgpoints']

            # Check if data for stereo calibration is valid 
            if len(self.imgs1) != len(self.imgs2) or len(objpts1) != len(objpts2):
                print('Invalid data !')
                return None

            self.stereo_params = {}
            
            # Setting for stereo calibration 
            flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH |
            cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |
            cv2.CALIB_FIX_K6)
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

            mtx1, dist1 = self.params1['mtx'], self.params1['dist']
            mtx2, dist2 = self.params2['mtx'], self.params2['dist']            

            
            ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpts1, imgpts1, imgpts2, mtx1, dist1, mtx2, dist2, self.img_size,
            criteria = criteria,
            flags=flags)

            self.stereo_params = {
                'ret':ret,
                'mtx1': M1,
                'dist1':d1,
                'mtx2': M2,
                'dist2':d2,
                'R':R,
                'T':T,
                'E':E,
                'F':F
            }

            self.stereo_calib_status = True

    def stereo_rectify(self, params = None, set_alpha=0):
        if self.stereo_params is not None:
            stereo = self.stereo_params
        else:
            if params is None:
                print('Parameters Unavailable')
                return None
            stereo = params

        M1, d1, M2, d2, R, T = stereo['mtx1'], stereo['dist1'], stereo['mtx2'], stereo['dist2'], stereo['R'], stereo['T']

        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(M1, d1, M2, d2, self.img_size, R, T, alpha = set_alpha)

        self.stereo_rectify_params = {
            'R1':R1,
            'R2':R2,
            'P1':P1,
            'P2':P2
        }

        return self.stereo_rectify_params

    def save_params(self, save_dir):
        if self.cam1_calib_status:
            np.save(save_dir + 'calibration1', self.params1)
        if self.cam2_calib_status:
            np.save(save_dir + 'calibration2', self.params2)
        if self.stereo_calib_status:
            np.save(save_dir + 'stereoCalibration', self.stereo_params)





