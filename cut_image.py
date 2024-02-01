import os
import cv2 as cv
import cv2
import detect_face as dface
import numpy as np

# path_face = r"E:\faces\20240118-114018.jpg"
# image = cv2.imread(path_face, -1)
# cv2.namedWindow("show", cv2.WINDOW_KEEPRATIO)
# cv2.imshow("show", image)
# cv2.waitKey(0)
#
# image1=image[10:20,-10:30]
# if image1.shape[0]==0 or image1.shape[1]==0:
#     print("none")
# cv2.imshow("show", image1)
# cv2.waitKey(0)
dir_path=r"E:\train_data\90000"

files=[os.path.join(dir_path,file) for file in os.listdir(dir_path)]
base_path=r"E:\train_data\etou\\"
for i,file in enumerate(files):
    print(file)
    image=cv.imread(file)
    # cv.imshow("show",image)
    # cv.waitKey(0)
    fea_all = dface.face_feature_all(image)
    if len(fea_all)==500:
        continue
    fea1_ind = np.asarray(range(len(fea_all)))
    # fea1_pix = dface.getfacefea_pix(fea_all, fea1_ind)
    rgb_fea1 = dface.getdot2pic(fea_all, fea1_ind, image)
    # rgb_fea1 = rgb_fea1  # [:,:,::-1]
    # cv2.namedWindow("result",cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("result", rgb_fea1)
    # cv2.waitKey(0)
    # 额头
    fea1_ind = np.asarray([151, 10, 333, 104])
    x1 = fea_all[fea1_ind[3]][0]
    x2 = fea_all[fea1_ind[2]][0]
    y2 = fea_all[fea1_ind[0]][1]
    y1 = int(y2 - 2.1 * (y2 - fea_all[fea1_ind[1]][1]))
    etou_image = image[y1:y2, x1:x2, :]
    # cv2.imshow("result",etou_image)
    # cv2.waitKey(0)
    if etou_image.shape[0] == 0 or etou_image.shape[1] == 0:
        continue
    cv.imwrite(base_path+str(i+12000)+".jpg",etou_image)