import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Load points for F estimation
def correspondPtsForF(ptsL_end, ptsR_end, ptsL_out, ptsR_out, idx_out, num, end_only=False):
    pts_F_left = []
    pts_F_right = []

    ptsL_end_col = []
    ptsR_end_col = []
    for i in ptsL_end.keys():
        ptL = ptsL_end[i]
        ptR = ptsR_end[i]
        ptsL_end_col.append(ptL)
        ptsR_end_col.append(ptR)

    ptsL_out_col = []
    ptsR_out_col = []
    for i in idx_out:
        if i in ptsL_out:
            if i in ptsR_out:
                ptL = ptsL_out[i]
                ptR = ptsR_out[i]
                ptsL_out_col.append(ptL)
                ptsR_out_col.append(ptR)
    
    if end_only:
        sampleNum = num
        pts_F_left = ptsL_end_col[0:sampleNum]
        pts_F_right = ptsR_end_col[0:sampleNum]
    else:
        sampleNum = int(num/2)
        pts_F_left = ptsL_end_col[0:sampleNum] + ptsL_out_col[0:sampleNum] 
        pts_F_right = ptsR_end_col[0:sampleNum] + ptsR_out_col[0:sampleNum] 
        
    return np.array(pts_F_left), np.array(pts_F_right)


def evaluate(pts_eval_L, pts_eval_R, F):    
    # Load points for evaluation

    pts_eval_L = np.array(pts_eval_L)
    pts_eval_R = np.array(pts_eval_R)

    # Left line ===> right points
    # Right line ===> left points
    linesL = cv2.computeCorrespondEpilines(pts_eval_L.reshape(-1, 1, 2), 1, F)
    linesR = cv2.computeCorrespondEpilines(pts_eval_R.reshape(-1, 1, 2), 2, F)

    linesL = linesL.reshape(-1, 3)
    linesR = linesR.reshape(-1, 3)

    pt_num = len(pts_eval_L)
    bias = np.ones((pt_num, 1))
    ptsL_bias = np.hstack((pts_eval_L, bias))
    ptsR_bias = np.hstack((pts_eval_R , bias))

    # Points distances
    resL = []
    resR = []

    for pts, line in zip(ptsR_bias, linesL):
        deno = line[0]**2 + line[1]**2
        res = np.dot(pts, line)**2/deno
        resR.append(res)

    for pts, line in zip(ptsL_bias, linesR):
        deno = line[0]**2 + line[1]**2
        res = np.dot(pts, line)**2/deno
        resL.append(res)

    # Normalize by number
    resL = np.array(resL)
    resR = np.array(resR)

    final_res = np.sum(resL + resR)/pt_num
    
    return final_res



if __name__ == "__main__":
    # Template points
    pts_eval_L = np.load('./calibration/blob_loc_left.npy')
    pts_eval_R = np.load('./calibration/blob_loc_right.npy')

    # End points
    ptsL_end = np.load('./data_F/ptsL800_end.npy').item()
    ptsR_end = np.load('./data_F/ptsR800_end.npy').item()

    # Joint Points
    ptsL_out = np.load('./data_F/ptsL800_out.npy').item()
    ptsR_out = np.load('./data_F/ptsR800_out.npy').item()

    idx_out = np.load('./data_F/eval_idx.npy')

    # Sample number
    sampleNum = list(np.linspace(10, 200, num=20).astype('int'))

    best_F = defaultdict(dict)

    res_LMEDS = defaultdict(dict)
    min_res_end = 10000
    min_res_mix = 10000
    # End Only
    for num in tqdm(sampleNum):
        # End points only
        pts_F_left, pts_F_right = correspondPtsForF(ptsL_end, ptsR_end, ptsL_out, ptsR_out, idx_out, num, end_only=True)
        F, mask = cv2.findFundamentalMat(pts_F_left, pts_F_right, cv2.FM_LMEDS)
        res = evaluate(pts_eval_L, pts_eval_R, F)
        res_LMEDS['end'][num] = res
        if res < min_res_end:
            min_res_end = res
            best_F['LMEDS']['end'] = [num, F, res]
        
        # Mix
        pts_F_left, pts_F_right = correspondPtsForF(ptsL_end, ptsR_end, ptsL_out, ptsR_out, idx_out, num)
        F, mask = cv2.findFundamentalMat(pts_F_left, pts_F_right, cv2.FM_LMEDS)
        res = evaluate(pts_eval_L, pts_eval_R, F)
        res_LMEDS['mix'][num] = res
        if res < min_res_mix:
            min_res_mix = res
            best_F['LMEDS']['mix'] = [num, F, res]

            
    res_RANSAC = defaultdict(dict)
    min_res_end = 10000
    min_res_mix = 10000
    # End Only
    for num in tqdm(sampleNum):
        # End points only
        pts_F_left, pts_F_right = correspondPtsForF(ptsL_end, ptsR_end, ptsL_out, ptsR_out, idx_out, num, end_only=True)
        F, mask = cv2.findFundamentalMat(pts_F_left, pts_F_right, cv2.FM_RANSAC)
        res = evaluate(pts_eval_L, pts_eval_R, F)
        res_RANSAC['end'][num] = res
        if res < min_res_end:
            min_res_end = res
            best_F['RANSAC']['end'] = [num, F, res]
        
        # Mix
        pts_F_left, pts_F_right = correspondPtsForF(ptsL_end, ptsR_end, ptsL_out, ptsR_out, idx_out, num)
        F, mask = cv2.findFundamentalMat(pts_F_left, pts_F_right, cv2.FM_RANSAC)
        res = evaluate(pts_eval_L, pts_eval_R, F)
        res_RANSAC['mix'][num] = res
        if res < min_res_mix:
            min_res_mix = res
            best_F['RANSAC']['mix'] = [num, F, res]

    res_8Pts = defaultdict(dict)
    min_res_end = 10000
    min_res_mix = 10000
    # End Only
    for num in tqdm(sampleNum):
        # End points only
        pts_F_left, pts_F_right = correspondPtsForF(ptsL_end, ptsR_end, ptsL_out, ptsR_out, idx_out, num, end_only=True)
        F, mask = cv2.findFundamentalMat(pts_F_left, pts_F_right, cv2.FM_8POINT)
        res = evaluate(pts_eval_L, pts_eval_R, F)
        res_8Pts['end'][num] = res
        if res < min_res_end:
            min_res_end = res
            best_F['8Pts']['end'] = [num, F, res]
        
        # Mix
        pts_F_left, pts_F_right = correspondPtsForF(ptsL_end, ptsR_end, ptsL_out, ptsR_out, idx_out, num)
        F, mask = cv2.findFundamentalMat(pts_F_left, pts_F_right, cv2.FM_8POINT)
        res = evaluate(pts_eval_L, pts_eval_R, F)
        res_8Pts['mix'][num] = res
        if res < min_res_mix:
            min_res_mix = res
            best_F['8Pts']['mix'] = [num, F, res]


    # All best F
    F0 = best_F['LMEDS']['end'][1]
    F1 = best_F['LMEDS']['mix'][1]

    F2 = best_F['RANSAC']['end'][1]
    F3 = best_F['RANSAC']['mix'][1]

    F4 = best_F['8Pts']['end'][1]
    F5 = best_F['8Pts']['mix'][1]

    

