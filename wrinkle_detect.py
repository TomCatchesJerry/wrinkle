import cv2 as cv
import cv2

import time
import numpy as np
from skimage.filters import frangi, gabor,hessian,meijering,sato
from skimage import measure, morphology,filters
import detect_face as dface

def testGabor(image):
    '''
    gabor滤波器对横向皱纹使用的参数frequency=0.085，theta=1.57
    theta与皱纹的方向有关，需要与皱纹方向一致
    @param image : 额头的图片
    @return: 皱纹列表,皱纹严重分数（0-100),越高越严重
    '''

    # cv2.namedWindow("result",cv2.WINDOW_KEEPRATIO)
    i_shape = image.shape
    image_o=image.copy()
    image = cv.resize(image, (500, 150))
    filter_w = image.shape[1] / 10
    # print(image.shape)
    image1 = image.copy()
    image2 = image.copy()
    image3 = image.copy()
    g = cv2.split(image1)[1]
    # frequency谐波函数的空间频率，以像素来表示,越小越能展示细节，theta滤波器方向

    # g=cv.blur(g,(10,4))
    # g = cv.blur(g, (5, 5))
    # cv2.imshow("result", g)
    # cv2.waitKey(0)
    # g = cv.medianBlur(g, 3)
    g = cv.blur(g, (3, 3))
    # cv2.imshow("result", g)
    # cv2.waitKey(0)
    # sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.084, theta = 1.57,sigma_x =7.0,sigma_y = 0.3)
    # sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.087, theta = 1.57, sigma_x = 6.1, sigma_y = 0.15)
    # sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.09, theta = 1.57,bandwidth = 1.5)
    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.09, theta = 1.57,bandwidth = 1.3)
    # cv2.imshow("result",sk_gabor_img_1)
    # cv2.waitKey(0)
    image1[sk_gabor_img_1>30]=255
    # cv2.imshow("result", image1)
    # cv2.waitKey(0)
    # cv2.imshow("result", sk_gabor_1)
    # cv2.waitKey(0)
    # image1[sk_gabor_1 > 30] = 255
    # cv2.imshow("result", image1)
    # cv2.waitKey(0)
    sk_gabor_img_1=sk_gabor_1
    sk_gabor_img_1[sk_gabor_img_1 < 30] = 0
    # 通过偏心率（>0.98）,连通区域面积来过滤一些非皱纹
    image_w = image.shape[1]
    bool_img = sk_gabor_img_1 > 30
    label_image = measure.label(bool_img)
    count = 0
    image_xcs = []
    for region in measure.regionprops(label_image):
        image_xc = np.zeros((image_w, 2))
        if region.area < 150:  # or region.area > 700
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        # print(region.orientation)
        if region.orientation>-np.pi/2.5 and region.orientation<np.pi/2.5:
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
        x = region.coords
        for i in range(len(x)):
            image_xc[x[i][1], 0] += 1
            image_xc[x[i][1], 1] += x[i][0]
        image_xc[image_xc[:, 0] == 0, 0] = 1
        # print(image_xc)
        image_xc[:, 1] = image_xc[:, 1] / image_xc[:, 0]
        image_xcs.append(image_xc.copy())

    wrikle_list = []
    for lis in image_xcs:
        wrikle = []
        for i in range(len(lis)):
            if lis[i][1] > 1:
                wrikle.append((i, lis[i][1]))
                image1[int(lis[i][1]), i] = 255
        if len(wrikle) == 0:
            continue
        wrikle_list.append(wrikle.copy())

    # 存放皱纹像素(x,y)列表的列表
    wrikle_list.sort(key = len, reverse = True)
    remove_list = []
    wrikle_list_ab = np.zeros((len(wrikle_list), 2, 2))
    for i in range(len(wrikle_list)):
        wrikle_list_ab[i, 0] = wrikle_list[i][0]
        wrikle_list_ab[i, 1] = wrikle_list[i][-1]

    # print(wrikle_list_ab)
    for i in range(len(wrikle_list) - 1):
        j = i + 1
        while j < len(wrikle_list):
            if wrikle_list_ab[j, 0, 0] > wrikle_list_ab[i, 0, 0] - 1 and wrikle_list_ab[j, 1, 0] < wrikle_list_ab[
                i, 1, 0] + 1 and \
                    abs(wrikle_list_ab[i, 0, 1] - wrikle_list_ab[j, 0, 1]) < 10 and abs(wrikle_list_ab[i, 1, 1] - wrikle_list_ab[j, 1, 1]) < 10:
                remove_list.append(j)
            j += 1
    remove_list.sort(reverse = True)
    # print(remove_list)
    remove_list = np.asarray(remove_list)
    remove_list = np.unique(remove_list)
    remove_list = remove_list[::-1]
    # print(remove_list)
    for i in remove_list:
        wrikle_list.pop(i)
        wrikle_list_ab = np.delete(wrikle_list_ab, i, axis = 0)

    # print(wrikle_list_ab)
    for wrik in wrikle_list:
        for (x, y) in wrik:
            image2[int(y), int(x)] = 255
        # cv2.imshow("result", image2)
        # cv2.waitKey(0)

    len1 = len(wrikle_list)

    #把链接皱纹的算法去掉
    # delete_list = []
    # for i in range(len1):
    #     j = i + 1
    #     while (j < len1):
    #         r1 = (wrikle_list_ab[i, 0, 0] - wrikle_list_ab[j, 1, 0]) ** 2 + (
    #                 wrikle_list_ab[i, 0, 1] - wrikle_list_ab[j, 1, 1]) ** 2
    #         if r1 < 300:
    #             wrikle_list_ab[i, 0] = wrikle_list_ab[j, 0]
    #             wrikle_list[i] = wrikle_list[i] + wrikle_list[j]
    #             delete_list.append(j)
    #         r2 = (wrikle_list_ab[i, 1, 0] - wrikle_list_ab[j, 0, 0]) ** 2 + (
    #                 wrikle_list_ab[i, 1, 1] - wrikle_list_ab[j, 0, 1]) ** 2
    #         if r2 < 300:
    #             wrikle_list_ab[i, 1] = wrikle_list_ab[j, 1]
    #             wrikle_list[i] = wrikle_list[i] + wrikle_list[j]
    #             delete_list.append(j)
    #         j += 1
    wrikle_list_final = []
    image3[:,:]=0
    for i in range(len(wrikle_list)):
        # if i in delete_list:
        #     continue

        wrik = wrikle_list[i]
        if len(wrik) < filter_w:
            continue
        wrikle_list_final.append(wrik)
        for (x, y) in wrik:
            if x < 1 and y < 1:
                continue

            if int(y) - 1 > 0:
                image3[int(y) - 1, int(x)] = 255
            # if int(y)-2>0:
            #     image3[int(y)-2,int(x)]=255
            image3[int(y), int(x)] = 255
            if int(y) + 1 < 150:
                image3[int(y) + 1, int(x)] = 255

    wrinkle_count = 0
    for wrinkle in wrikle_list:
        wrinkle_count += len(wrinkle)
    # print(wrinkle_count/400/500.0)
    wrinkle_level = wrinkle_count / (image.shape[0] * image.shape[1]) / 0.09 * 100
    # print(wrinkle_count/(image.shape[0]*image.shape[0]))
    if wrinkle_level > 100:
        wrinkle_level = 100
    image3 = cv2.resize(image3, (i_shape[1], i_shape[0]))
    # print(image_o.shape)
    # print(image3.shape)
    image_o[image3>100]=255
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
    print(im_s)
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
        if region.eccentricity > 0.9:
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
    i_shape=image.shape
    image_o=image.copy()
    c_shape=(200,300)
    image=cv.resize(image,c_shape)
    filter_w=image.shape[1]/10
    image1=image.copy()
    # image2=image.copy()
    image3=image.copy()
    # cv2.imshow("result", image)
    # cv2.waitKey(0)
    g = cv2.split(image1)[1]
    g=cv.blur(g,(4,4))

    # print("g shape: ",g.shape)
    #左脸颊的theta = 0.68，右脸颊的theta = -0.68
    the=0.68
    if left==False:
        the=-0.68
    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.085, theta = the,sigma_x =7.0,sigma_y =1.3)
    # sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency = 0.08, theta = the,bandwidth = 1.1)


    # sk_frangi_img2 = frangi(g, sigmas = (3, 10, 2), alpha = 0.5,
    #                         beta = 0.95, gamma = 1.2)
    # sk_frangi_img2 = hessian(g, sigmas = (1, 5, 2), alpha = 0.5,
    #                         beta = 0.95, gamma = 1.0)
    # sk_frangi_img2=sato(g,sigmas = [2.5])
    # cv2.imshow("result", sk_gabor_img_1)
    # cv2.waitKey(0)
    # cv2.imshow("result", sk_gabor_1)
    # cv2.waitKey(0)
    # sk_frangi_img2[sk_frangi_img2>0.1]=255
    # cv2.imshow("result", sk_frangi_img2)
    # cv2.waitKey(0)
    # sk_frangi_img2[sk_frangi_img2 > 0.1] = 255
    # cv2.imshow("result", sk_frangi_img2)
    # cv2.waitKey(0)
    # sk_gabor_img_1=sk_gabor_1
        # cv2.imshow("result", sk_gabor_1)
        # cv2.waitKey(0)
    # return
    # sk_gabor_img_1[sk_gabor_img_1 > 0.10] = 255
    # sk_gabor_img_1[sk_gabor_img_1 <= 0.10] = 0
    sk_gabor_img_1[sk_gabor_img_1<30]=0
    # cv2.imshow("result", sk_gabor_img_1)
    # cv2.waitKey(0)
    # sk_gabor_img_1=cv.dilate(sk_gabor_img_1,(15,5))
    # cv2.imshow("result", sk_gabor_img_1)
    # cv2.waitKey(0)
    #通过偏心率（>0.98）,连通区域面积来过滤一些非皱纹
    image_w=image.shape[1]
    # print(image_w)

    bool_img = sk_gabor_img_1>30
    label_image = measure.label(bool_img)
    count = 0
    image_xcs=[]
    for region in measure.regionprops(label_image):
        image_xc = np.zeros((image_w, 2))
        if region.area < 150:  # or region.area > 700
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        if region.eccentricity > 0.90:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                sk_gabor_img_1[x[i][0]][x[i][1]] = 0
            continue
        x = region.coords
        for i in range(len(x)):
            image_xc[x[i][1],0]+=1
            image_xc[x[i][1],1]+=x[i][0]
        image_xc[image_xc[:,0]==0, 0]=1
        # print(image_xc)
        image_xc[:,1]=image_xc[:,1]/image_xc[:,0]
        image_xcs.append(image_xc.copy())
    wrikle_list=[]
    for lis in image_xcs:
        wrikle=[]
        for i in range(len(lis)):
            if lis[i][1] > 1:
                wrikle.append((i,lis[i][1]))
                image1[int(lis[i][1]),i]=255
        if len(wrikle) == 0:
            continue
        wrikle_list.append(wrikle.copy())
        # cv2.imshow("result", image1)
        # cv2.waitKey(0)
    # #存放皱纹像素(x,y)列表的列表
    # wrikle_list.sort(key=len,reverse = True)
    # remove_list=[]
    # wrikle_list_ab = np.zeros((len(wrikle_list),2,2))
    # for i in range(len(wrikle_list)):
    #     wrikle_list_ab[i,0]=wrikle_list[i][0]
    #     wrikle_list_ab[i,1]=wrikle_list[i][-1]

    # print(wrikle_list_ab)
    # for i in range(len(wrikle_list)-1):
    #     j=i+1
    #     while j<len(wrikle_list):
    #         if wrikle_list_ab[j,0,0]>wrikle_list_ab[i,0,0]-1 and wrikle_list_ab[j,1,0] < wrikle_list_ab[i,1,0] + 1 and \
    #                 abs(wrikle_list_ab[i, 0, 1]-wrikle_list_ab[j, 0, 1] )<20 and abs(wrikle_list_ab[i, 1, 1]-wrikle_list_ab[j, 1, 1] )<20 :
    #             remove_list.append(j)
    #         j+=1
    # remove_list.sort(reverse = True)
    # # print(remove_list)
    # remove_list=np.asarray(remove_list)
    # remove_list=np.unique(remove_list)
    # remove_list=remove_list[::-1]
    # # print(remove_list)
    # for i in remove_list:
    #     wrikle_list.pop(i)
    #     wrikle_list_ab=np.delete(wrikle_list_ab,i,axis=0)

    # print(wrikle_list_ab)
    # for wrik in wrikle_list:
    #     for (x,y) in wrik:
    #         image2[int(y),int(x)]=255
    #     cv2.imshow("result", image2)
    #     cv2.waitKey(0)

    image3[:,:]=0
    wrikle_list_final=[]
    for i in range(len(wrikle_list)):
        wrik=wrikle_list[i]
        # if len(wrik)<filter_w:
        #     continue
        wrikle_list_final.append(wrik)
        for (x,y) in wrik:
            if x<1 and y<1:
                continue

            if int(y)-1>0:
                image3[int(y)-1,int(x)]=255
            if int(y)-2>0:
                image3[int(y)-2,int(x)]=255
            image3[int(y),int(x)]=255
            if int(y)+1<c_shape[1]:

                image3[int(y)+1,int(x)]=255
            if int(y) + 2 < c_shape[1]:
                image3[int(y)+2,int(x)]=255
        # cv2.imshow("result", image3)
        # cv2.waitKey(0)
    # 通过将同一x的所有像素拟合出一个值的方式去细化皱纹
    # 如何把多段皱纹连成一整条
    # 目前的想法是先找到每条小皱纹的两端，
    # 然后以端点为中心去进行一定半径内的圆搜索
    """
    cv2.imshow("result", sk_gabor_img_1)
    cv2.waitKey(0)
    skel1 = morphology.skeletonize(sk_gabor_img_1)
    print(skel1)
    image1=image.copy()
    image[skel1 ] = 255
    cv2.imshow("result", image)
    cv2.waitKey(0)
    """
    wrinkle_count=0
    for wrinkle in wrikle_list:
        wrinkle_count+=len(wrinkle)
    # print(wrinkle_count/400/500.0)
    wrinkle_level=wrinkle_count/(image.shape[0]*image.shape[1])/0.06*100
    # print(wrinkle_count/(image.shape[0]*image.shape[0]))
    if wrinkle_level>100:
        wrinkle_level=100
    image3=cv2.resize(image3,(i_shape[1],i_shape[0]))
    image_o[image3==255]=255
    return image_o, 100-int(wrinkle_level)


def detect_wrinkle_using_mediapipe(image):
    """
    传入人脸图片，使用mediapipe进行特征点检测，然后根据特征点分割出人脸的
    额头、左右眼角、左右脸颊，然后分别带入函数检测皱纹，然后在原图上画上皱纹
    :param image: 人脸图片
    :return: 带皱纹的人脸图片，一个皱纹分数数组，包含五个数值(额头皱纹分数，左眼角皱纹分数，右眼角皱纹分数，
    左法令纹分数，右法令纹分数)
    """
    fea_all=dface.face_feature_all(image)
    # fea1_ind = np.asarray(range(len(fea_all)))
    # fea1_pix = dface.getfacefea_pix(fea_all, fea1_ind)
    # rgb_fea1 = dface.getdot2pic(fea_all, fea1_ind, image)
    # rgb_fea1 = rgb_fea1  # [:,:,::-1]
    # cv2.namedWindow("result",cv2.WINDOW_KEEPRATIO)
    #额头
    fea1_ind = np.asarray([151, 10, 333, 104])
    x1 = fea_all[fea1_ind[3]][0]
    x2 = fea_all[fea1_ind[2]][0]
    y2 = fea_all[fea1_ind[0]][1]
    y1 = int(y2 - 2.1 * (y2 - fea_all[fea1_ind[1]][1]))
    etou_image = image[y1:y2, x1:x2, :]
    # cv2.imshow("result",etou_image)
    # cv2.waitKey(0)
    # cv2.imwrite(r"E:\faces\face\1.jpg",etou_image)
    etou_image,level1=testGabor(etou_image)
    image[y1:y2, x1:x2]=etou_image

    #左眼角
    lyanjiao_ind = np.asarray([21,227,127,226])
    lyanjiao_list=np.asarray([71,70,46,113,226,111,116])#21,,227
    lyanjiao_pix=dface.getfacefea_pix(fea_all, lyanjiao_list)
    x1=fea_all[lyanjiao_ind[2]][0]
    x2=fea_all[lyanjiao_ind[3]][0]
    y1=fea_all[lyanjiao_ind[0]][1]
    y2=fea_all[lyanjiao_ind[1]][1]
    lyanjiao_pix[:,0]=lyanjiao_pix[:,0]-x1
    lyanjiao_pix[:,1]=lyanjiao_pix[:,1]-y1
    lyanjiao_image = image[y1:y2, x1:x2, :]
    # cv2.imshow("result", lyanjiao_image)
    # cv2.waitKey(0)
    # cv2.imwrite(r"E:\faces\face\2.jpg", lyanjiao_image)
    lyanjiao_image,level2=detect_canthus_wrinkle(lyanjiao_image,lyanjiao_pix)
    image[y1:y2, x1:x2, :]=lyanjiao_image
    # rgb_fea1 = dface.getdot2pic(fea_all, lyanjiao_ind, image)
    # rgb_fea1 = dface.getdot2pic(fea_all, lyanjiao_list, rgb_fea1)

    #右眼角
    ryanjiao_ind = np.asarray([251,454,356,446])
    ryanjiao_list = np.asarray([301,300,276,342,446,340,345])#251,454
    ryanjiao_pix=dface.getfacefea_pix(fea_all, ryanjiao_list)
    x1 = fea_all[ryanjiao_ind[3]][0]
    x2 = fea_all[ryanjiao_ind[2]][0]
    y1 = fea_all[ryanjiao_ind[0]][1]
    y2 = fea_all[ryanjiao_ind[1]][1]
    ryanjiao_image = image[y1:y2, x1:x2, :]
    # cv2.imshow("result", ryanjiao_image)
    # cv2.waitKey(0)
    # cv2.imwrite( r"E:\faces\face\3.jpg",ryanjiao_image)
    ryanjiao_pix[:,0]=ryanjiao_pix[:,0]-x1
    ryanjiao_pix[:,1]=ryanjiao_pix[:,1]-y1
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

    # path = r"E:\faces\face\5.jpg"
    # path = r"E:\SU\sources\Wrinkles_detection\wrinkle\muouwen\8.PNG"
    # # path = r"E:\SU\sources\Wrinkles_detection\realtest\2.jpg"
    # image = cv2.imread(path, -1)
    # image1 ,_ = detect_no_wrinkle(image,True)
    # cv2.namedWindow("show", cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("show", image1)
    # cv2.waitKey(0)



    for i in range(8):
        path_face = r"E:\faces\\"+str(27+1)+".jpg"
        image = cv2.imread(path_face, -1)
        cv2.namedWindow("show", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("show", image)
        cv2.waitKey(0)
        image1,lev=detect_wrinkle_using_mediapipe(image)
        cv2.imshow("show", image1)
        cv2.waitKey(0)
        print(lev)
    # # #test3
    # # path = "8.jpg"
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
