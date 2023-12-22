import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

def face_feature_all(pic):
    # pic是cv2读入的图，BGR三通道
    # 最后返回是468点（先列后行)
    rgb = pic.copy()
    (h,w) = (np.shape(rgb)[0],np.shape(rgb)[1])
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9)
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        calface_dec = face_mesh.process(rgb)
    # 得到人脸边界
    fadot = calface_dec.multi_face_landmarks[0]
    fa_dot = np.zeros((len(fadot.landmark), 2))

    for i in range(len(fadot.landmark)):
        fa_dot[i, 0], fa_dot[i, 1] = float(w * fadot.landmark[i].x), h * float(fadot.landmark[i].y)
    fea_pix = fa_dot.astype(np.int32)
    return fea_pix

def getfacefea_pix(facpix,ind):
    #ind是468点中选出特征点的索引
    #facpix是468点的集合
    #输出是对应索引位置的像素点
    pix = facpix[ind,:]
    pix = pix[pix[:,0]>0,:]
    pix = pix[pix[:, 1] > 0, :]
    # pix1 = pix.astype(np.int32)
    return pix

def getdot2pic(fa_dot,ind,pic):
    # 用于可视化的函数
    # fa_dot是特征点
    # ind是待取的索引
    # pic是三通道的图片形式
    if np.max(pic) <=1:
        pic = np.ceil(pic*255)
        pic = pic.astype(np.uint8)
    dotind = fa_dot[ind,:]
    # dotind = dotind[:,::-1]
    dotpic = pic.copy()
    for i in dotind:
        cv2.circle(dotpic, i, 5, (0, 255, 0), 10)
    return dotpic

# 以下输入图片，返回特征点
if __name__ == "__main__":
    # 载入人脸，需是完整的正面人脸
    for i in range(8):

        path_face = r"E:\faces\\"+str(i+1)+".jpg"

        rgb_ini = cv2.imread(path_face,-1)
        cv2.namedWindow("show",cv2.WINDOW_KEEPRATIO)
        cv2.imshow("show",rgb_ini)
        cv2.waitKey(0)
        # rgb_ini_r90 = cv2.rotate(rgb_ini, cv2.ROTATE_90_CLOCKWISE)
        rgb = rgb_ini#[:,:,::-1]

        fea_all = face_feature_all(rgb)
        fea1_ind = np.asarray([151,10,333,104])
        x1=fea_all[fea1_ind[3]][0]
        x2=fea_all[fea1_ind[2]][0]
        y2=fea_all[fea1_ind[0]][1]
        y1=y2-2*(y2-fea_all[fea1_ind[1]][1])
        rgb_ini1=rgb_ini[y1:y2,x1:x2,:]
        # print(len(fea_all))
        fea1_ind=np.asarray(range(len(fea_all)))
        fea1_pix = getfacefea_pix(fea_all, fea1_ind)
        rgb_fea1 = getdot2pic(fea_all,fea1_ind,rgb)
        rgb_fea1=rgb_fea1#[:,:,::-1]
        cv2.imshow("show", rgb_fea1)
        cv2.waitKey(0)
        save_path=r"E:\forehead\\"+str(i+1)+".jpg"
        # cv2.imwrite(save_path,rgb_ini1)