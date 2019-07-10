index = 73
# be careful about coordinate representation
sample_num = 200

# Find points to be triangulated in left image 
img_left = cv2.imread('../segmentation/seg_result/seg' + str(index) + '_r.png')
img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

# A conversion in coordinate is needed 
start = time.time()
coordy_left, coordx_left = np.where(img_left == 255)
pts_dict_L = {}
points_left = []
for x, y in zip(coordx_left, coordy_left):
    pts_dict_L[y] = x
    #points_left.append([x, y])
if sample_num > len(coordy_left):
    print('Sampel number exceeed total points number!')
    
sample_idx = np.floor(np.linspace(min(coordy_left), max(coordy_left), num = sample_num))
for ele in sample_idx:
    if ele in pts_dict_L:
        points_left.append([pts_dict_L[ele], ele])

points_left = np.array(points_left).astype('float32')
end = time.time()
print(end - start)

# Find points to be triangulated in right image 
img_right = cv2.imread('../segmentation/seg_result/seg' + str(index) + '_l.png')
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

coordy_right, coordx_right = np.where(img_right == 255)
points_right = []
for x, y in zip(coordx_right, coordy_right):
    points_right.append([x,y])

points_right = np.array(points_right).astype('float32')



bias = np.ones((len(points_right), 1))
points_right_bias = np.hstack((points_right, bias))
points_right_bias


lines = cv2.computeCorrespondEpilines(points_left.reshape(-1, 1, 2), 1, F)
lines = lines.reshape(-1,3)


# Obtain all matched points in right image
# By calculating distance of point to line
thresh = 0.5
ptLineRes = np.abs(np.matmul(lines, points_right_bias.T))
# All the corresponding poinst found by index
min_idx = np.argmin(ptLineRes, axis=1)
min_arg = np.min(ptLineRes, axis=1)
# Filter them by distance thresh
th_idx = np.where(min_arg < thresh)
ft_idx = np.take(min_idx, th_idx)
points_rt = np.take(points_right, ft_idx, axis = 0)
points_lt = np.take(points_left, th_idx, axis = 0)