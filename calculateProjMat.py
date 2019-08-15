import cv2
import numpy as np


coeff_left = np.load('./calibration/calibration_coeff_left.npy').item()
kl = coeff_left['intr_mat']

coeff_right = np.load('./calibration/calibration_coeff_right.npy').item()
kr = coeff_right['intr_mat']

trian_param = np.load('./calibration/triangulate_coeff.npy').item()

F = trian_param['F']

# Estimate essential matrix
E = np.matmul(np.matmul(kr.T, F), kl)
U, S, Vt = np.linalg.svd(E)
# Tweak S
diag = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 0]])
E_new = np.matmul(np.matmul(U, diag), Vt)
Un, Sn, Vtn = np.linalg.svd(E_new)
# Define W and Wt
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
Wt = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

# 2 possible solutions for Rotation and Translation
R1 = np.matmul(np.matmul(Un, W), Vtn)
R2 = np.matmul(np.matmul(Un, Wt), Vtn)
T1 = Un[:,2].reshape(3,1)
T2 = -Un[:,2].reshape(3,1)

# 4 Possible projection matrix for right camera
PR1 = np.hstack((R1, T1))
PR2 = np.hstack((R1, T2))
PR3 = np.hstack((R2, T1))
PR4 = np.hstack((R2, T2))

# Initialize Left projection matrix
PL = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])