import cv2 as cv
import cv2

import time
import numpy as np
from skimage.filters import frangi, gabor,hessian,meijering,sato
from skimage import measure, morphology,filters
import detect_face as dface

#本文件与后端皱纹检测算法同步

def testGabor(image):
    '''
    gabor滤波器对横向皱纹使用的参数frequency=0.085，theta=1.57
    theta与皱纹的方向有关，需要与皱纹方向一致
    @param image : 额头的图片
    @return: 皱纹列表,皱纹严重分数（0-100),越高越严重
    '''

    # cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)
    i_shape = image.shape
    image_o = image.copy()
    # image_w=1000
    image_w = i_shape[1]
    # image_h=300
    image_h = i_shape[0]
    # g=cv2.split(image)[0]
    # image = cv.resize(image, (image_w, image_h))
    # filter_w = image.shape[1] / 200
    filter_w = image_w / 12.5
    # print(image.shape)
    image1 = image.copy()
    g = cv2.split(image1)[0]
    # frequency谐波函数的空间频率，以像素来表示,越小越能展示细节，theta滤波器方向
    # cv2.imshow("result", g)
    # cv2.waitKey(0)
    g = cv.blur(g, (3, 3))
    # cv2.imshow("result", g)
    # cv2.waitKey(0)

    g = g.astype(float)
    # max_v = np.max(g)
    # min_v = np.min(g)
    # g[:, :] = (g[:, :] - min_v) * 255.0 / (max_v - min_v)
    mean_v = np.mean(g)
    factor = 0.4
    g = g.astype(float)
    g[:, :] = (g[:, :] - mean_v) * factor + g[:, :]
    g[g > 255] = 255
    g[g < 0] = 0
    g = g.astype(np.uint8)
    # cv2.imshow("result", g)
    # cv2.waitKey(0)
    mask_g = g.copy()
    mask_g[:, :] = 0
    # g = cv.resize(g, (image_w, image_h))
    # 比较有用的参数修改sigmas，gama
    # sk_frangi_img2 = frangi(g, sigmas = (3, 10, 2.5), alpha = 0.5, beta = 0.95, gamma = 1.2)
    sk_frangi_img = g.copy()
    sk_frangi_img[:, :] = 0
    for the in (1.27, 1.57, 1.87):

        sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.084, theta = the, sigma_x = 6.2)  # 1.77,1.37
        # sk_gabor_img_2, sk_gabor_2 = gabor(g, frequency = 0.084, theta = 1.27, sigma_x = 6.2)#1.77,1.37
        # sk_gabor_img_3, sk_gabor_3 = gabor(g, frequency = 0.084, theta = 1.87, sigma_x = 6.2)#1.77,1.37
        sk_frangi_img2 = sk_gabor_img_1
        # cv2.imshow("result1", sk_frangi_img2)
        # cv2.waitKey(0)
        # cv2.imshow("result2", sk_gabor_img_2)
        # cv2.waitKey(0)
        # cv2.imshow("result3", sk_gabor_img_3)
        # cv2.waitKey(0)
        thres = 30
        sk_frangi_img2[sk_frangi_img2 > thres] = 255
        # sk_gabor_img_2[sk_gabor_img_2 > 30] = 255
        # sk_gabor_img_3[sk_gabor_img_3 > 30] = 255
        sk_frangi_img2[sk_frangi_img2 <= thres] = 0
        # sk_frangi_img2[sk_gabor_img_2==255]=255
        # sk_frangi_img2[sk_gabor_img_3==255]=255

        # cv2.imshow("result", sk_frangi_img2)
        # cv2.waitKey(0)
        image1[sk_frangi_img2 == 255] = 255
        # cv2.imshow("result1", image1)
        # cv2.waitKey(0)
        # sk_gabor_img_1=sk_frangi_img2
        # sk_gabor_img_1[sk_gabor_img_1 < 30] = 0
        # 通过偏心率（>0.98）,连通区域面积来过滤一些非皱纹
        image_w = image.shape[1]
        bool_img = sk_frangi_img2 > 30
        label_image = measure.label(bool_img)
        count = 0
        image_xcs = []
        # thres=image.shape[0]*image.shape[1]/8000
        for region in measure.regionprops(label_image):
            image_xc = np.zeros((image_w, 2))
            # if region.area < thres:  # or region.area > 700
            #     x = region.coords
            #     for i in range(len(x)):
            #         sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            #     continue
            # print(region.axis_major_length)
            if region.axis_major_length < filter_w:
                x = region.coords
                for i in range(len(x)):
                    sk_frangi_img2[x[i][0]][x[i][1]] = 0
                continue
            # print(region.feret_diameter_max)
            # print(region.axis_major_length)
            # print(region.orientation)
            if region.orientation > -np.pi / 3 and region.orientation < np.pi / 3:
                x = region.coords
                for i in range(len(x)):
                    sk_frangi_img2[x[i][0]][x[i][1]] = 0
                continue
            if region.eccentricity > 0.98:
                count += 1
            else:
                x = region.coords
                for i in range(len(x)):
                    sk_frangi_img2[x[i][0]][x[i][1]] = 0
                continue
            x = region.coords
            for i in range(len(x)):
                image_xc[x[i][1], 0] += 1
                image_xc[x[i][1], 1] += x[i][0]
            image_xc[image_xc[:, 0] == 0, 0] = 1
            # print(image_xc)
            image_xc[:, 1] = image_xc[:, 1] / image_xc[:, 0]
            image_xcs.append(image_xc.copy())
        sk_frangi_img[(sk_frangi_img == 255) | (sk_frangi_img2 == 255)] = 255
        wrikle_list = []
        for lis in image_xcs:
            wrikle = []
            for i in range(len(lis)):
                if lis[i][1] > 1:
                    wrikle.append((i, lis[i][1]))
                    # image1[int(lis[i][1]), i] = 255
            # if len(wrikle) == 0:
            #     continue
            wrikle_list.append(wrikle.copy())
        wrikle_list_final = []
        sk_frangi_img2[:, :] = 0
        for i in range(len(wrikle_list)):
            # if i in delete_list:
            #     continue

            wrik = wrikle_list[i]
            # if len(wrik) < filter_w:
            #     continue
            wrikle_list_final.append(wrik)
            for (x, y) in wrik:
                if x < 1 and y < 1:
                    continue

                if int(y) - 1 > 0:
                    mask_g[int(y) - 1, int(x)] = 255
                # if int(y)-2>0:
                #     image3[int(y)-2,int(x)]=255
                mask_g[int(y), int(x)] = 255
                if int(y) + 1 < image_h:
                    mask_g[int(y) + 1, int(x)] = 255

    wrinkle_count = 0
    # for wrinkle in wrikle_list:
    #     wrinkle_count += len(wrinkle)
    # print(wrinkle_count/400/500.0)
    # cv2.imshow("r1", mask_g)
    # cv2.waitKey(0)
    # wrinkle_count = np.count_nonzero(mask_g)
    wrinkle_count = np.count_nonzero(sk_frangi_img)
    wrinkle_level = wrinkle_count / (image.shape[0] * image.shape[1]) / 0.11 * 100
    # print(wrinkle_count / (image.shape[0] * image.shape[1]))
    if wrinkle_level > 100:
        wrinkle_level = 100
    # mask_g = cv2.resize(mask_g, (i_shape[1], i_shape[0]))
    # sk_frangi_img = cv2.resize(sk_frangi_img, (i_shape[1], i_shape[0]))
    # print(image_o.shape)
    # print(image3.shape)
    image_o[mask_g > 100] = 255
    return image_o, 100 - int(wrinkle_level)
def face_wrinkle(image,area=0):
    if area==0:
        return testGabor(image)

def draw_wrinkle(image,wrinkle_image,offset=[0,0]):

    """

    :param image: 图片
    :param wrinkle_list: 皱纹
    :param offset: 偏置(x,y)
    :return: 画了皱纹的图片
    """

    w_s = wrinkle_image.shape
    image[offset[1]:offset[1] + w_s[0], offset[0]:offset[0] + w_s[1]] = wrinkle_image
    # print(image.shape)
    return image

def detect_canthus_wrinkle(image,canthus_list=[]):
    '''
    检测眼角皱纹的接口
    :param image: 待检测的眼角的图片
    :param canthus_list: 眼角检测区域点，[(p1x,p1y),···],这些点是在人脸识别点在眼角图片内的位置
    :return: 带有皱纹的眼角图片，皱纹等级（0-100），分数越高，皱纹越少
    '''
    # cv2.namedWindow("result",cv2.WINDOW_KEEPRATIO)
    im_s = image.shape
    image_o=image.copy()
    # print(im_s)
    image = cv2.resize(image, (100, 200))
    image1 = cv2.split(image)[0]
    # 滤波的尺度，较小可以提取较细的，较大可以提取较粗的
    # alpha是调整检测片状结构的灵敏度参数，越小，灵敏度越高
    # beta是检测斑点状结构的灵敏度参数。越小，灵敏度越高
    # sk_frangi_img2 = frangi(image1, sigmas = (3, 10, 2), alpha = 0.5,
    #                         beta = 0.70)  # 线宽范围，步长，连接程度（越大连接越多），减少程度(越大减得越多)0.015
    # sk_frangi_img2 = frangi(image1, sigmas = (3, 10, 2), alpha = 0.5,
    #                         beta = 0.70, gamma = 1.5)
    sk_frangi_img2 = frangi(image1, sigmas = (3, 10, 2), alpha = 0.5, beta = 0.95, gamma = 1.2)
    # cv2.imshow("result", sk_frangi_img2)
    # cv2.waitKey(0)

    # sk_frangi_img2[sk_frangi_img2>0.02]=255
    # sk_frangi_img2[sk_frangi_img2<=0.02]=0
    sk_frangi_img2[sk_frangi_img2 > 0.15] = 255
    sk_frangi_img2[sk_frangi_img2 <= 0.15] = 0
    # cv2.imshow("result", sk_frangi_img2)
    # cv2.waitKey(0)
    # return
    image_w = image.shape[1]
    label_image = measure.label(sk_frangi_img2)
    count = 0
    image_xcs = []
    for region in measure.regionprops(label_image):
        image_xc = np.zeros((image_w, 2))
        image1 = image.copy()
        if region.area < 50:  # or region.area > 700
            x = region.coords
            for i in range(len(x)):
                sk_frangi_img2[x[i][0]][x[i][1]] = 0
            continue
        # 后需的偏心率需要调大些
        if region.eccentricity > 0.8:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                sk_frangi_img2[x[i][0]][x[i][1]] = 0
            continue
        x = region.coords
        for i in range(len(x)):
            image1[x[i][0]][x[i][1]] = 255
        # cv2.imshow("result", image1)
        # cv2.waitKey(0)
        for i in range(len(x)):
            image_xc[x[i][1], 0] += 1
            image_xc[x[i][1], 1] += x[i][0]
        image_xc[image_xc[:, 0] == 0, 0] = 1
        # print(image_xc)
        image_xc[:, 1] = image_xc[:, 1] / image_xc[:, 0]
        image_xcs.append(image_xc.copy())
    # cv2.imshow("result", sk_frangi_img2)
    # cv2.waitKey(0)
    # skel = morphology.skeletonize(sk_frangi_img2)
    mask1=sk_frangi_img2.copy()
    mask1[:,:]=0
    mask1 = cv2.resize(mask1, (im_s[1], im_s[0]))
    if len(canthus_list)!=0:
        mask1 = cv.fillPoly(mask1, [np.array(canthus_list)], (255))
    else:
        mask1[:,:]=255
    # cv2.imshow("result", mask1)
    # cv2.waitKey(0)
    sk_frangi_img2 = cv2.resize(sk_frangi_img2, (im_s[1], im_s[0]))
    sk_frangi_img2[mask1!=255]=0
    skel = sk_frangi_img2 == 255
    # image2 = image.copy()
    # image2 = cv2.resize(image2, (im_s[1], im_s[0]))
    image_o[skel] = 255
    wrinkle_count = np.count_nonzero(skel)
    # print(wrinkle_count / (image.shape[0] * image.shape[1]))
    wrinkle_level = wrinkle_count / (image.shape[0] * image.shape[1]) / 0.13 * 100
    if wrinkle_level > 100:
        wrinkle_level = 100

    # cv2.imshow("result", image2)
    # cv2.waitKey(0)
    return image_o, int(100 - wrinkle_level)

def detect_no_wrinkle(image,left = True):
    '''

    检测法令纹
    gabor滤波器对横向皱纹使用的参数
    theta与皱纹的方向有关，需要与皱纹方向一致
    @param image : 额头的图片
    @param left 是否为左脸颊图片，右脸颊图片则left=False
    @return: 皱纹列表,皱纹严重分数（0-100),越高越严重
    '''

    # cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)
    # i_shape=image.shape
    # print(i_shape)
    image_o = image.copy()
    filter_w = image.shape[0] / 5
    image1 = image.copy()
    # cv2.imshow("result", image)
    # cv2.waitKey(0)
    g = cv2.split(image1)[0]
    # cv2.imshow("result", g)
    # cv2.waitKey(0)
    g = g.astype(float)
    # max_v = np.max(g)
    # min_v = np.min(g)
    # g[:, :] = (g[:, :] - min_v) * 255.0 / (max_v - min_v)
    mean_v = np.mean(g)
    factor = 0.6
    g = g.astype(float)
    g[:, :] = (g[:, :] - mean_v) * factor + g[:, :]
    g[g > 255] = 255
    g[g < 0] = 0
    g = g.astype(np.uint8)
    # cv2.imshow("result", g)
    # cv2.waitKey(0)
    # print("g shape: ",g.shape)
    # 左脸颊的theta = 0.68，右脸颊的theta = -0.68
    the = 0.68
    if left == False:
        the = -0.68
    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.08, theta = the, bandwidth = 1.1)
    sk_gabor_img_1[sk_gabor_img_1 > 30] = 255
    # cv2.imshow("result", sk_gabor_img_1)
    # cv2.waitKey(0)
    sk_gabor_img_1[sk_gabor_img_1 < 30] = 0
    # 通过偏心率（>0.98）,连通区域面积来过滤一些非皱纹
    bool_img = sk_gabor_img_1 > 30
    label_image = measure.label(bool_img)
    count = 0
    for region in measure.regionprops(label_image):
        if region.axis_major_length < filter_w:  # or region.area > 700
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        if region.eccentricity > 0.98:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue

    wrinkle_count = np.count_nonzero(sk_gabor_img_1)
    # print(wrinkle_count/400/500.0)
    wrinkle_level = wrinkle_count / (image.shape[0] * image.shape[1]) / 0.08 * 100
    # print(wrinkle_count/(image.shape[0]*image.shape[0]))
    if wrinkle_level > 100:
        wrinkle_level = 100
    # image3=cv2.resize(sk_gabor_img_1,(i_shape[1],i_shape[0]))
    image_o[sk_gabor_img_1 == 255] = 255
    return image_o, 100 - int(wrinkle_level)

#放大图片检测更能检测出细小的皱纹
def testGabor1(image):
    '''

    多尺度检测
    1000*400 and 500*150
    大尺度的情况下可以检测出教轻微的皱纹
    gabor滤波器对横向皱纹使用的参数frequency=0.085，theta=1.57
    theta与皱纹的方向有关，需要与皱纹方向一致
    @param image : 额头的图片
    @return: 皱纹列表,皱纹严重分数（0-100),越高越严重
    '''

    cv2.namedWindow("result",cv2.WINDOW_KEEPRATIO)
    i_shape = image.shape
    image_o=image.copy()
    image_w=1000
    image_h=400
    image = cv.resize(image, (image_w, image_h))
    # filter_w = image.shape[1] / 200
    filter_w = image_w / 12.5
    # print(image.shape)
    image1 = image.copy()
    g = cv2.split(image1)[1]
    # frequency谐波函数的空间频率，以像素来表示,越小越能展示细节，theta滤波器方向
    g = cv.medianBlur(g, 3)
    # g = cv.blur(g, (3, 3))
    cv2.imshow("result", g)
    cv2.waitKey(0)
    #比较有用的参数修改sigmas，gama
    sk_frangi_img2 = frangi(g, sigmas = (3, 10, 2.5), alpha = 0.5, beta = 0.95, gamma = 1.2)
    cv2.imshow("result1", sk_frangi_img2)
    cv2.waitKey(0)
    thres=0.10
    sk_frangi_img2[sk_frangi_img2 > thres] = 255
    sk_frangi_img2[sk_frangi_img2 <= thres] = 0
    cv2.imshow("result", sk_frangi_img2)
    cv2.waitKey(0)
    image1[sk_frangi_img2==255]=255
    cv2.imshow("result1", image1)
    cv2.waitKey(0)
    # sk_gabor_img_1=sk_frangi_img2
    # sk_gabor_img_1[sk_gabor_img_1 < 30] = 0
    # 通过偏心率（>0.98）,连通区域面积来过滤一些非皱纹
    image_w = image.shape[1]
    bool_img = sk_frangi_img2 > 30
    label_image = measure.label(bool_img)
    count = 0
    image_xcs = []
    # thres=image.shape[0]*image.shape[1]/8000
    for region in measure.regionprops(label_image):
        image_xc = np.zeros((image_w, 2))
        # if region.area < thres:  # or region.area > 700
        #     x = region.coords
        #     for i in range(len(x)):
        #         sk_gabor_img_1[x[i][0]][x[i][1]] = 0
        #     continue
        # print(region.axis_major_length)
        if region.axis_major_length < filter_w:
            x = region.coords
            for i in range(len(x)):
                sk_frangi_img2[x[i][0]][x[i][1]] = 0
            continue
        print(region.feret_diameter_max)
        print(region.axis_major_length)
        # print(region.orientation)
        if region.orientation>-np.pi/3 and region.orientation<np.pi/3:
            x = region.coords
            for i in range(len(x)):
                sk_frangi_img2[x[i][0]][x[i][1]] = 0
            continue
        if region.eccentricity > 0.98:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                sk_frangi_img2[x[i][0]][x[i][1]] = 0
            continue
        x = region.coords
        for i in range(len(x)):
            image_xc[x[i][1], 0] += 1
            image_xc[x[i][1], 1] += x[i][0]
        image_xc[image_xc[:, 0] == 0, 0] = 1
        # print(image_xc)
        image_xc[:, 1] = image_xc[:, 1] / image_xc[:, 0]
        image_xcs.append(image_xc.copy())
    sk_frangi_img=sk_frangi_img2.copy()
    wrikle_list = []
    for lis in image_xcs:
        wrikle = []
        for i in range(len(lis)):
            if lis[i][1] > 1:
                wrikle.append((i, lis[i][1]))
                # image1[int(lis[i][1]), i] = 255
        # if len(wrikle) == 0:
        #     continue
        wrikle_list.append(wrikle.copy())
    wrikle_list_final = []
    sk_frangi_img2[:,:]=0
    for i in range(len(wrikle_list)):
        # if i in delete_list:
        #     continue

        wrik = wrikle_list[i]
        # if len(wrik) < filter_w:
        #     continue
        wrikle_list_final.append(wrik)
        for (x, y) in wrik:
            if x < 1 and y < 1:
                continue

            if int(y) - 1 > 0:
                sk_frangi_img2[int(y) - 1, int(x)] = 255
            # if int(y)-2>0:
            #     image3[int(y)-2,int(x)]=255
            sk_frangi_img2[int(y), int(x)] = 255
            if int(y) + 1 < image_h:
                sk_frangi_img2[int(y) + 1, int(x)] = 255

    wrinkle_count = 0
    for wrinkle in wrikle_list:
        wrinkle_count += len(wrinkle)
    # print(wrinkle_count/400/500.0)
    wrinkle_count=np.count_nonzero(sk_frangi_img2)
    wrinkle_level = wrinkle_count / (image.shape[0] * image.shape[1]) / 0.13 * 100
    print(wrinkle_count/(image.shape[0]*image.shape[1]))
    if wrinkle_level > 100:
        wrinkle_level = 100
    sk_frangi_img2 = cv2.resize(sk_frangi_img2, (i_shape[1], i_shape[0]))
    sk_frangi_img = cv2.resize(sk_frangi_img, (i_shape[1], i_shape[0]))
    # print(image_o.shape)
    # print(image3.shape)
    image_o[sk_frangi_img2>100]=255
    return image_o, 100 - int(wrinkle_level)
def detect_wrinkle_using_mediapipe(image):
    """
    传入人脸图片，使用mediapipe进行特征点检测，然后根据特征点分割出人脸的
    额头、左右眼角、左右脸颊，然后分别带入函数检测皱纹，然后在原图上画上皱纹
    :param image: 人脸图片
    :return: 带皱纹的人脸图片，一个皱纹分数数组，包含五个数值(额头皱纹分数，左眼角皱纹分数，右眼角皱纹分数，
    左法令纹分数，右法令纹分数)
    """
    fea_all=dface.face_feature_all(image)
    fea1_ind = np.asarray(range(len(fea_all)))
    # fea1_pix = dface.getfacefea_pix(fea_all, fea1_ind)
    rgb_fea1 = dface.getdot2pic(fea_all, fea1_ind, image)
    if len(fea_all)==500:
        return image,(0,0,0,0,0)
    # rgb_fea1 = rgb_fea1  # [:,:,::-1]
    # cv2.namedWindow("result",cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("result", rgb_fea1)
    # cv2.waitKey(0)
    #额头
    fea1_ind = np.asarray([151, 10, 333, 104])
    x1 = fea_all[fea1_ind[3]][0]
    x2 = fea_all[fea1_ind[2]][0]
    y2 = fea_all[fea1_ind[0]][1]
    y1 = int(y2 - 2.1 * (y2 - fea_all[fea1_ind[1]][1]))
    if y1<=0:
        return image, (0, 0, 0, 0, 0)
    etou_image = image[y1:y2, x1:x2, :]
    if etou_image.shape[0]==0 or etou_image.shape[1]==0:
        return image, (0, 0, 0, 0, 0)
# cv2.imshow("result",etou_image)
    # cv2.waitKey(0)
    # cv2.imwrite(r"E:\faces\face\1.jpg",etou_image)
    etou_image,level1=testGabor(etou_image)
    image[y1:y2, x1:x2]=etou_image

    #左眼角
    lyanjiao_ind = np.asarray([21,227,127,226])
    lyanjiao_list=np.asarray([71,70,46,113,226,111,116])#21,,227
    lyanjiao_list1=np.asarray([139,156,124])#21,,227
    lyanjiao_pix=dface.getfacefea_pix(fea_all, lyanjiao_list)
    lyanjiao_pix1=dface.getfacefea_pix(fea_all, lyanjiao_list1)
    x1=fea_all[lyanjiao_ind[2]][0]
    x2=fea_all[lyanjiao_ind[3]][0]
    y1=fea_all[lyanjiao_ind[0]][1]
    y2=fea_all[lyanjiao_ind[1]][1]
    # print(lyanjiao_pix)
    # print(lyanjiao_pix1)
    lyanjiao_pix[0:3]=(lyanjiao_pix[0:3]+lyanjiao_pix1[0:3])/2
    # print(lyanjiao_pix)
    lyanjiao_pix[:,0]=lyanjiao_pix[:,0]-x1
    lyanjiao_pix[:,1]=lyanjiao_pix[:,1]-y1
    lyanjiao_image = image[y1:y2, x1:x2, :]
    # cv2.imshow("result", lyanjiao_image)
    # cv2.waitKey(0)
    # cv2.imwrite(r"E:\faces\face\2.jpg", lyanjiao_image)
    if lyanjiao_image.shape[0]==0 or lyanjiao_image.shape[1]==0:
        return image, (0, 0, 0, 0, 0)
    lyanjiao_image,level2=detect_canthus_wrinkle(lyanjiao_image,lyanjiao_pix)
    image[y1:y2, x1:x2, :]=lyanjiao_image
    # rgb_fea1 = dface.getdot2pic(fea_all, lyanjiao_ind, image)
    # rgb_fea1 = dface.getdot2pic(fea_all, lyanjiao_list, rgb_fea1)

    #右眼角
    ryanjiao_ind = np.asarray([251,454,356,446])
    ryanjiao_list = np.asarray([301,300,276,342,446,340,345])#251,454
    ryanjiao_list1 = np.asarray([368,383,353])#251,454
    ryanjiao_pix=dface.getfacefea_pix(fea_all, ryanjiao_list)
    ryanjiao_pix1=dface.getfacefea_pix(fea_all, ryanjiao_list1)
    x1 = fea_all[ryanjiao_ind[3]][0]
    x2 = fea_all[ryanjiao_ind[2]][0]
    y1 = fea_all[ryanjiao_ind[0]][1]
    y2 = fea_all[ryanjiao_ind[1]][1]
    ryanjiao_image = image[y1:y2, x1:x2, :]
    # cv2.imshow("result", ryanjiao_image)
    # cv2.waitKey(0)
    # cv2.imwrite( r"E:\faces\face\3.jpg",ryanjiao_image)
    ryanjiao_pix[0:3] = (ryanjiao_pix[0:3] + ryanjiao_pix1[0:3]) / 2
    ryanjiao_pix[:,0]=ryanjiao_pix[:,0]-x1
    ryanjiao_pix[:,1]=ryanjiao_pix[:,1]-y1
    if ryanjiao_image.shape[0]==0 or ryanjiao_image.shape[1]==0:
        return image, (0, 0, 0, 0, 0)
    ryanjiao_image,level3=detect_canthus_wrinkle(ryanjiao_image,ryanjiao_pix)
    image[y1:y2, x1:x2, :]=ryanjiao_image
    # rgb_fea1 = dface.getdot2pic(fea_all, ryanjiao_ind, rgb_fea1)
    # rgb_fea1 = dface.getdot2pic(fea_all, ryanjiao_list, rgb_fea1)

    #左脸颊
    llianjia_ind = np.asarray([187,101,165,57])
    # rgb_fea1 = dface.getdot2pic(fea_all, llianjia_ind, rgb_fea1)
    x1 = fea_all[llianjia_ind[0]][0]
    x2 = fea_all[llianjia_ind[2]][0]
    y1 = fea_all[llianjia_ind[1]][1]
    y2 = fea_all[llianjia_ind[3]][1]
    llianjia_image = image[y1:y2, x1:x2, :]
    # cv2.imshow("result", llianjia_image)
    # cv2.waitKey(0)
    # cv2.imwrite(r"E:\faces\face\4.jpg",llianjia_image)
    # ryanjiao_pix = dface.getfacefea_pix(fea_all, ryanjiao_list)
    if llianjia_image.shape[0] == 0 or llianjia_image.shape[1] == 0:
        return image, (0, 0, 0, 0, 0)
    llianjia_image,level4=detect_no_wrinkle(llianjia_image)
    image[y1:y2, x1:x2, :]=llianjia_image

    #右脸颊
    rlianjia_ind = np.asarray([411, 330, 391,287])
    # rgb_fea1 = dface.getdot2pic(fea_all, rlianjia_ind, rgb_fea1)
    x1 = fea_all[rlianjia_ind[2]][0]
    x2 = fea_all[rlianjia_ind[0]][0]
    y1 = fea_all[rlianjia_ind[1]][1]
    y2 = fea_all[rlianjia_ind[3]][1]
    rlianjia_image = image[y1:y2, x1:x2, :]
    # cv2.imshow("result", rlianjia_image)
    # cv2.waitKey(0)
    # cv2.imwrite(r"E:\faces\face\5.jpg",rlianjia_image)
    if rlianjia_image.shape[0] == 0 or rlianjia_image.shape[1] == 0:
        return image, (0, 0, 0, 0, 0)
    rlianjia_image,level5=detect_no_wrinkle(rlianjia_image,False)
    image[y1:y2, x1:x2, :]=rlianjia_image
    # cv2.imshow("result", image)
    # cv2.waitKey(0)

    #画法令纹
    lfaling_ind = np.asarray([203,206,216])
    rfaling_ind = np.asarray([423,426,436])
    lfaling_pix=dface.getfacefea_pix(fea_all, lfaling_ind)
    rfaling_pix=dface.getfacefea_pix(fea_all, rfaling_ind)
    cv2.line(image,lfaling_pix[0],lfaling_pix[1],(255,255,255),3)
    cv2.line(image,lfaling_pix[1],lfaling_pix[2],(255,255,255),3)
    cv2.line(image,rfaling_pix[0],rfaling_pix[1],(255,255,255),3)
    cv2.line(image,rfaling_pix[1],rfaling_pix[2],(255,255,255),3)
    # cv2.imshow("result", image)
    # cv2.waitKey(0)

    return image,(level1,level2,level3,level4,level5)
    pass


if __name__ == '__main__':

    # path = r"E:\faces\face\1.jpg"
    # # path = r"E:\SU\sources\Wrinkles_detection\wrinkle\muouwen\8.PNG"
    # # # path = r"E:\SU\sources\Wrinkles_detection\realtest\2.jpg"
    # image = cv2.imread(path, -1)
    # # image1 ,_ = detect_no_wrinkle(image,True)
    # image1 ,_ = testGabor(image)
    # cv2.namedWindow("show", cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("show", image1)
    # cv2.waitKey(0)

    # path_face = r"E:\faces\20240118-100104.jpg"
    path_face = r"E:\faces\20240118-114018.jpg"
    image = cv2.imread(path_face, -1)
    cv2.namedWindow("show", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("show", image)
    cv2.waitKey(0)
    image1, lev = detect_wrinkle_using_mediapipe(image)
    cv2.imshow("show", image1)
    cv2.waitKey(0)
    print(lev)

    # for i in range(7):
    #     path_face = r"E:\faces\\"+str(33+i)+".jpg"
    #     # path_face = r"E:\faces\20240116-155210.jpg"
    #     image = cv2.imread(path_face, -1)
    #     cv2.namedWindow("show", cv2.WINDOW_KEEPRATIO)
    #     cv2.imshow("show", image)
    #     cv2.waitKey(0)
    #     image1,lev=detect_wrinkle_using_mediapipe(image)
    #     cv2.imshow("show", image1)
    #     cv2.waitKey(0)
    #     print(lev)

    # # #test3
    # # path = "8.jpg
    # path=r"E:\SU\sources\Wrinkles_detection\wrinkle\yuweiwen\8.PNG"
    # path = r"E:\forehead\8.jpg"
    # image = cv2.imread(path, -1)
    # image2 = image[0:, 0:]
    # cv2.namedWindow("show", cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("show", image2)
    # cv2.waitKey(0)
    # # image3, wl = detect_canthus_wrinkle4(image2)
    # # image3,wl=detect_canthus_wrinkle2(image2)
    # print(wl)
    # image4 = draw_canthus_wrinkle(image, image3, (0, 0))
    # cv2.imshow("show", image4)
    # cv2.waitKey(0)



    # # #test2
    # path=r"E:\SU\sources\Wrinkles_detection\8.jpg"
    # # path=r"E:\faces\9.jpg"
    # path=r"E:\SU\sources\Wrinkles_detection\wrinkle\yuweiwen\1.PNG"
    # image=cv2.imread(path,-1)
    # image2=image[0:,0:]
    # cv2.namedWindow("show",cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("show",image2)
    # cv2.waitKey(0)
    # image3,wl=detect_canthus_wrinkle(image2)#,[(0,0),(0,166),(128,166),(128,0)]
    # # image3,wl=detect_canthus_wrinkle2(image2)
    # print(wl)
    # image4=draw_wrinkle(image,image3,(0,0))
    # cv2.imshow("show",image4)
    # cv2.waitKey(0)


    # test1 face_wrinkle testGabor draw_wrinkle
    # base_path = r"E:\SU\sources\Wrinkles_detection\realtest\\"
    # base_path = r"E:\forehead\\"
    # for i in range(8):
    #     index = 1 + i
    #     index = str(index)
    #     path = base_path + index + ".jpg"
    #     # path="5.jpg"
    #     # path="2885.png"
    #     image = cv2.imread(path, -1)
    #     wrinkle_lise, level = face_wrinkle(image)
    #     # wrinkle_lise, level = testGabor1(image)
    #     print(level)
    #     image1 = draw_wrinkle(image, wrinkle_lise)
    #     cv.imshow("show", image1)
    #     cv.waitKey(0)
